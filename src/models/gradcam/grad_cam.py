"""Extract saliency using Grad-CAM

Reference: https://github.com/jacobgil/pytorch-grad-cam
"""
import torch
import torch.nn.functional as F


class GradCam():
    def __init__(self, device, model, feature_module, target_layer_names=None):
        self.device = device
        self.model = model
        self.feature_module = feature_module
        self.extractor = _ModelOutputs(self.model, self.feature_module, target_layer_names)

    def __call__(self, inputs, indices=None, resize=True):
        """Get the saliency masks from a batch of inputs

        Args:
            inputs: input tensor of shape (B, C, H, W)
            indices: indices of classes to backprop from. If None, pick the
                classes with the highest scores
            resize: whether to resize the output to input shape

        Return:
            cam: saliency masks of shape (B, H', W') where (H', W') is the size
                of target feature. If resize is True, the masks are resized to
                (B, H, W).
        """
        inputs.requires_grad_(True)
        batch_size = len(inputs)

        # Switch model to eval st dropout and batch_norm work in eval mode
        was_training = self.model.training
        if was_training:
            self.model.eval()

        # Extract feature and output
        features, outputs = self.extractor(inputs)

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

        # Global average pooling
        weights = grads_val.mean(dim=(2, 3))

        # Cum sum across channel dimension
        cam_shape = [batch_size, target.shape[2], target.shape[3]]
        cam = torch.zeros(cam_shape, dtype=torch.float32).to(self.device)
        for i in range(weights.shape[1]):
            cam += weights[:, i].unsqueeze(-1).unsqueeze(-1) * target[:, i, :, :]

        # Postprocess cam
        cam = torch.clamp(cam, 0)
        if resize:
            cam = cam.unsqueeze(dim=1)
            cam = F.interpolate(cam, size=inputs.shape[2:],
                                mode='bilinear', align_corners=False)
            cam = cam[:, 0, :, :]
        cam = cam - cam.min()
        cam = cam / cam.max()

        # Switch back to the previous mode
        if was_training:
            self.model.train()

        return cam


class _FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        # If the model does not have modules
        if len(self.model._modules) == 0:
            x = self.model(x)
            x.register_hook(self.save_gradient)
            outputs += [x]
            return outputs, x

        # The model has multiple modules
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class _ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = _FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x
