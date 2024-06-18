from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb


##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            # get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks


def get_multi_blocks(num_layers):
    if num_layers == 50:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3),
            # get_block(in_channel=64, depth=128, num_units=4),
            # get_block(in_channel=128, depth=256, num_units=14),
            # get_block(in_channel=256, depth=512, num_units=3)
        ]
        blocks2 = [
            # get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            # get_block(in_channel=128, depth=256, num_units=14),
            # get_block(in_channel=256, depth=512, num_units=3)
        ]
        blocks3 = [
            # get_block(in_channel=64, depth=64, num_units=3),
            # get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            # get_block(in_channel=256, depth=512, num_units=3)
        ]

    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks1, blocks2, blocks3


class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir', img_channel=3):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(img_channel, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = F.interpolate(x, size=112)
        x = self.input_layer(x)
        x = self.body(x)
        # x = self.output_layer(x)
        # return l2_norm(x)

        return x.view(-1, 49, 1024)


class MultiBackbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(MultiBackbone, self).__init__()
        # assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks1, blocks2, blocks3 = get_multi_blocks(num_layers)
        # blocks2 = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules1 = []
        for block in blocks1:
            for bottleneck in block:
                modules1.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        modules2 = []
        for block in blocks2:
            for bottleneck in block:
                modules2.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        modules3 = []
        for block in blocks3:
            for bottleneck in block:
                modules3.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        # modules4 = []
        # for block in blocks4:
        #     for bottleneck in block:
        #         modules4.append(
        #             unit_module(bottleneck.in_channel,
        #                         bottleneck.depth,
        #                         bottleneck.stride))
        self.body1 = Sequential(*modules1)
        self.body2 = Sequential(*modules2)
        self.body3 = Sequential(*modules3)
        # self.body4 = Sequential(*modules4)

    def forward(self, x):
        x = F.interpolate(x, size=112)
        x = self.input_layer(x)
        x1 = self.body1(x)
        x2 = self.body2(x1)
        x3 = self.body3(x2)

        # x = self.output_layer(x)
        # return l2_norm(x)

        return x3, x2, x1

# import collections
#
# model1 = Backbone(50, 0.0, 'ir')
# model2 = MultiBackbone(50, 0.0, 'ir')
#
# ir_checkpoint = torch.load(r'D:\PythonProject\FER\models\pretrain\ir50.pth',
#                            map_location=lambda storage, loc: storage)
# ir_checkpoint_new = torch.load(r'D:\PythonProject\FER\models\pretrain\new_ir50.pth',
#                            map_location=lambda storage, loc: storage)
# state_dict1 = model1.state_dict()
# state_dict2 = model2.state_dict()
# new_state_dict = collections.OrderedDict()
#
# for (k1, v1), (k2, v2), (k3, v3) in zip(state_dict1.items(), ir_checkpoint.items(),state_dict2.items()):
#     print(f'k1:{k1}, k2:{k2}, k3:{k3}')
#     if k1 == k2:
#         new_state_dict[k3] = v2
#
# torch.save(new_state_dict, r'D:\PythonProject\FER\models\pretrain\new_ir50.pth')
