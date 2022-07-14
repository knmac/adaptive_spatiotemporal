"""Extract layers a model from start_layer to stop_layer

Reference: tf_model_zoo/bninception/pytorch_load.py
"""
import os
import sys

import torch
import yaml
from torch import nn

from tf_model_zoo.bninception.layer_factory import get_basic_layer, parse_expr


_MODEL_PATH = 'tf_model_zoo/bninception/bn_inception.yaml'
_WEIGHT_URL = 'https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth'


class SubBNInception(nn.Module):
    """Split a model and return from start_layer to stop_layer"""

    def __init__(self, model_path=_MODEL_PATH, weight_url=_WEIGHT_URL,
                 start_layer=None, stop_layer=None):
        super(SubBNInception, self).__init__()
        self.start_layer = start_layer
        self.stop_layer = stop_layer

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        # Read model config from yaml file
        if not os.path.isfile(model_path):
            model_path = os.path.join(root, model_path)

        with open(model_path, 'r') as stream:
            try:
                manifest = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(-1)

        # Build model architecture from yaml file content
        layers = manifest['layers']
        self._channel_dict = dict()
        self._op_list = list()

        for layer in layers:
            out_var, op, in_var = parse_expr(layer['expr'])
            if op != 'Concat':
                channels = 3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]]
                (id,
                 out_name,
                 module,
                 out_channel,
                 in_name) = get_basic_layer(info=layer, channels=channels, conv_bias=True)
                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel

        # Load weights
        try:
            state_dict = torch.utils.model_zoo.load_url(weight_url)
        except PermissionError:
            model_dir = os.path.join(root, '.cache/torch/checkpoints')
            state_dict = torch.utils.model_zoo.load_url(weight_url, model_dir=model_dir)

        for k, v in state_dict.items():
            state_dict[k] = torch.squeeze(v, dim=0)

        self.load_state_dict(state_dict)

        self._extract_layers()

    def _extract_layers(self):
        """Extract layers wrt start_layer and stop_layer"""
        start_layer, stop_layer = self.start_layer, self.stop_layer

        # Assign layer name if not specified
        if (start_layer is None) and (stop_layer is None):
            # Skip everything if no start or stop layers specified
            self.start_layer = self._op_list[0][0]
            self.stop_layer = self._op_list[-1][0]
            return
        start_layer = self._op_list[0][0] if start_layer is None else start_layer
        stop_layer = self._op_list[-1][0] if stop_layer is None else stop_layer
        self.start_layer, self.stop_layer = start_layer, stop_layer

        # Check the layers
        valid_start, valid_stop = -1, -1
        for i, item in enumerate(self._op_list):
            layer_id = item[0]
            if layer_id == start_layer:
                valid_start = i
            if layer_id == stop_layer:
                valid_stop = i
            if valid_start >= 0 and valid_stop >= 0:
                break
        assert valid_start >= 0, 'start_layer not found'
        assert valid_stop >= 0, 'stop_layer not found'
        assert valid_start <= valid_stop, 'start_layer must be before stop_layer'

        # Remove layers
        tmp_lst = [x for x in self._op_list]
        for i, item in enumerate(tmp_lst):
            if i < valid_start or i > valid_stop:
                if item[1] != 'Concat':
                    id, op, out_name, in_name = item
                    if out_name in self._channel_dict:
                        del self._channel_dict[out_name]
                    delattr(self, id)
                    self._op_list.remove(item)
                else:
                    id, op, out_var_0, in_var = item
                    self._op_list.remove(item)
                    if out_var_0 in self._channel_dict:
                        del self._channel_dict[out_var_0]

    def forward(self, input):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except Exception:
                    for x in op[-1]:
                        print(x, data_dict[x].size())
                    raise
        return data_dict[self._op_list[-1][2]]


def test():
    full = SubBNInception()
    first_half = SubBNInception(stop_layer='pool2_3x3_s2')
    second_half = SubBNInception(start_layer='inception_3a_1x1')

    full_op_list = full._op_list
    first_op_list = first_half._op_list
    second_op_list = second_half._op_list

    first_op_list.extend(second_op_list)
    # assert full_op_list == first_op_list
    if full_op_list == first_op_list:
        print('Same op lists')

    full_channel_dict = full._channel_dict
    first_channel_dict = first_half._channel_dict
    second_channel_dict = second_half._channel_dict

    first_channel_dict.update(second_channel_dict)
    # assert full_channel_dict == first_channel_dict
    if full_channel_dict == first_channel_dict:
        print('Same channel dicts')

    full_attr = set(dir(full))
    first_attr = set(dir(first_half))
    second_attr = set(dir(second_half))

    first_attr.update(second_attr)
    # assert full_attr == first_attr
    if full_attr == first_attr:
        print('Same attributes')


if __name__ == '__main__':
    test()
