"""GradCam for BNInception. Overload the extractor from GradCam because the
forward is different.
"""
import torch

from .grad_cam import GradCam


class GradCamBNInception(GradCam):
    def __init__(self, device, model, feature_module):
        self.device = device
        self.model = model
        self.feature_module = feature_module
        self.extractor = _BNInceptionOutputs(self.model, self.feature_module)


class _BNInceptionOutputs():
    """For BNInception the feature_module is the name of the output layer
    because some layers functions are not defined as attributes
    """
    def __init__(self, model, feature_module):
        self.model = model
        self.feature_module = feature_module
        self.gradients = []

        assert isinstance(feature_module, str)
        found = False
        for op in model._op_list:
            if feature_module == op[2]:
                found = True
                break
        assert found is True, 'feature_module not found'

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_activations = []
        data_dict = dict()
        data_dict[self.model._op_list[0][-1]] = x

        for op in self.model._op_list:
            if (op[1] != 'Concat') and (op[1] != 'InnerProduct'):
                data_dict[op[2]] = getattr(self.model, op[0])(data_dict[op[-1]])
            elif op[1] == 'InnerProduct':
                tmp = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self.model, op[0])(tmp.view(tmp.shape[0], -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[k] for k in op[-1]), 1)
                except Exception:
                    for k in op[-1]:
                        print(k, data_dict[k].size())
                    raise

            # Add a hook at the layer to extract gradient
            if op[2] == self.feature_module:
                data_dict[op[2]].register_hook(self.save_gradient)
                target_activations = [data_dict[op[2]]]
        return target_activations, data_dict[self.model._op_list[-1][2]]
