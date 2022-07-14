"""GradCAM for the pipeline because the forwarding and feature modules are
different
"""
import torch
import torch.nn.functional as F


class GradCamPipeline():
    def __init__(self, device, model, component, chosen_modality, output_type, feature_module):
        self.device = device
        self.model = model
        self.chosen_modality = chosen_modality
        self.output_type = output_type
        self.extractor = _PipelineOutputs(model, component, chosen_modality, feature_module)

    def __call__(self, inputs, indices=None, keep_negative=False, resize=True, new_size=None):
        """Get the saliency masks from a batch of inputs

        Args:
            inputs: dictionary of modalities; each item is input tensor of
                shape (B, T*C, H, W), where T = number of frames, C = channel
                dim of each frame
            indices: indices of classes to backprop from. If None, pick the
                classes with the highest scores
            keep_negative: whether to keep negative values of GradCAM. If False,
                will remove them (ReLU)
            resize: whether to resize the output to input shape
            new_size: only valid if resize is True. If new_size is None, reuse
                the input size, otherwise use new_size

        Return:
            cam: saliency masks of shape (B, T, H', W') where (H', W') is the
                size of target feature. If resize is True, the masks are
                resized to (B, T, H, W).
            target: target feature of shape (B, T, D, H', W'), where D is the
                feature dimension
        """
        for k in inputs:
            inputs[k].requires_grad_(True)
            batch_size = len(inputs[k])
        num_segments = self.model.num_segments

        # Switch model to eval st dropout and batch_norm work in eval mode
        was_training = self.model.training
        if was_training:
            self.model.eval()

        # Extract feature and output
        features, outputs = self.extractor(inputs)
        outputs = outputs[self.output_type]

        # Set index as the highest class if not provided
        if indices is None:
            indices = torch.argmax(outputs, dim=1)
        assert len(indices) == batch_size

        # Retrieve the scores wrt to indices
        one_hot = torch.zeros(outputs.shape, dtype=torch.float32).to(self.device)
        for i in range(batch_size):
            one_hot[i, indices[i]] = 1
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * outputs)

        # Clean up gradients
        # self.feature_module.zero_grad()
        self.model.zero_grad()

        # Compute gradient by backprop
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1]  # last feature

        # Get the target feature
        target = features[-1]

        # Recover batch_size and num_segments
        t_shape = [batch_size, num_segments, target.shape[1], target.shape[2], target.shape[3]]
        g_shape = [batch_size, num_segments, grads_val.shape[1], grads_val.shape[2], grads_val.shape[3]]
        target = target.reshape(t_shape)
        grads_val = grads_val.reshape(g_shape)

        # Global average pooling
        weights = grads_val.mean(dim=(2, 3))

        # Cum sum across chanel dimension
        cam_shape = [batch_size, num_segments, target.shape[3], target.shape[4]]
        cam = torch.zeros(cam_shape, dtype=torch.float32).to(self.device)
        for i in range(weights.shape[1]):
            cam += weights[:, :, i].unsqueeze(-1).unsqueeze(-1) * target[:, :, i, :, :]

        # Postprocess cam
        if not keep_negative:
            cam = torch.clamp(cam, 0)
        if resize:
            if new_size is None:
                new_size = inputs[self.chosen_modality].shape[2:]
            cam = F.interpolate(cam, size=new_size,
                                mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()

        # Switch back to the previous mode
        if was_training:
            self.model.train()

        return cam, target


class _PipelineOutputs():
    def __init__(self, pipeline, component, chosen_modality, feature_module):
        self.pipeline = pipeline
        self.component = component
        self.chosen_modality = chosen_modality
        self.feature_module = feature_module
        self.gradients = []

        if component != pipeline.light_model:
            print('Only support the first part of the pipline for now')
            raise NotImplementedError

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_activations = {}

        # Light model
        concatenated = []
        for m in self.pipeline.modality:
            if (m == 'RGB'):
                channel = 3
            elif (m == 'Flow'):
                channel = 2
            elif (m == 'Spec'):
                channel = 1
            sample_len = channel * self.pipeline.light_model.new_length[m]

            if m == 'RGBDiff':
                sample_len = 3 * self.pipeline.light_model.new_length[m]
                input[m] = self.pipeline.light_model._get_diff(input[m])

            base_model = getattr(self.pipeline.light_model, m.lower())
            x_in = x[m].view((-1, sample_len) + x[m].size()[-2:])

            # Forward as bninception model
            target_activations[m], base_out = self._forward_bninception(
                x_in, base_model, m)
            base_out = base_out.view(base_out.size(0), -1)
            concatenated.append(base_out)
        target_activations = target_activations[self.chosen_modality]
        x = torch.cat(concatenated, dim=1)

        # Time sampler and heavy model
        x = self.pipeline.time_sampler(x, self.pipeline.heavy_model)

        # Space sampler
        x = self.pipeline.space_sampler(x)

        # Action recognition model
        x = self.pipeline.actreg_model(x)

        x = {'verb': x[0], 'noun': x[1]}
        return target_activations, x

    def _forward_bninception(self, x, model, current_modality):
        """Forward function for bninception model"""
        target_activations = []
        data_dict = dict()
        data_dict[model._op_list[0][-1]] = x

        # Check the feature_module
        found = False
        for op in model._op_list:
            if self.feature_module == op[2]:
                found = True
                break
        assert found is True, 'feature_module not found'

        # Forwarding
        for op in model._op_list:
            if (op[1] != 'Concat') and (op[1] != 'InnerProduct'):
                data_dict[op[2]] = getattr(model, op[0])(data_dict[op[-1]])
            elif op[1] == 'InnerProduct':
                tmp = data_dict[op[-1]]
                data_dict[op[2]] = getattr(model, op[0])(tmp.view(tmp.shape[0], -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[k] for k in op[-1]), 1)
                except Exception:
                    for k in op[-1]:
                        print(k, data_dict[k].size())
                    raise

            # Add a hook at the layer to extract gradient
            if (op[2] == self.feature_module) and (self.chosen_modality == current_modality):
                data_dict[op[2]].register_hook(self.save_gradient)
                target_activations = [data_dict[op[2]]]
        return target_activations, data_dict[model._op_list[-1][2]]
