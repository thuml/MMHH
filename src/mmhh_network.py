import numpy as np
from collections import OrderedDict

import torch

import torch.nn as nn
from torchvision import models
import math
import torch.nn.functional as F

"""
Refer to https://github.com/thuml/HashNet 
"""


class AlexNetFc(nn.Module):
    def __init__(self, logger, hash_bit, increase_scale=True):
        """
        :param hash_bit: output hash bit
        :param increase_scale: if the scale increase gradually. True for HashNet, False for DHN
        """
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        hash_layer = nn.Linear(model_alexnet.classifier[6].in_features, hash_bit)
        hash_layer.weight.data.normal_(0, 0.01)
        hash_layer.bias.data.fill_(0.0)
        self.hash_layer = hash_layer
        self.__in_features = hash_bit
        self.activation = nn.Tanh()

        # HashNet part
        self.iter_num = 0
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.scale = self.init_scale
        self.increase_scale = increase_scale
        logger.info("increase_scale is %s" % increase_scale)

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        y = self.hash_layer(x)
        if self.increase_scale and self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y

    def get_parameter_list(self):
        return [{"params": self.feature_layers.parameters(), "lr": 1},
                {"params": self.hash_layer.parameters(), "lr": 10}]

    def output_num(self):
        return self.__in_features


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNetFc(nn.Module):
    def __init__(self, logger, name, hash_bit, increase_scale=True):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[name](pretrained=True)
        conv1 = model_resnet.conv1
        bn1 = model_resnet.bn1
        relu = model_resnet.relu
        maxpool = model_resnet.maxpool
        layer1 = model_resnet.layer1
        layer2 = model_resnet.layer2
        layer3 = model_resnet.layer3
        layer4 = model_resnet.layer4
        avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(conv1, bn1, relu, maxpool,
                                            layer1, layer2, layer3, layer4, avgpool)
        hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        hash_layer.weight.data.normal_(0, 0.01)
        hash_layer.bias.data.fill_(0.0)
        self.hash_layer = hash_layer
        self.__in_features = hash_bit
        self.activation = nn.Tanh()

        # HashNet part
        self.iter_num = 0
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.scale = self.init_scale
        self.increase_scale = increase_scale
        logger.info("increase_scale is %s" % increase_scale)

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        if self.increase_scale and self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y

    def get_parameter_list(self):
        return [{"params": self.feature_layers.parameters(), "lr": 1},
                {"params": self.hash_layer.parameters(), "lr": 10}]

    def output_num(self):
        return self.__in_features


vgg_dict = {"VGG11": models.vgg11, "VGG13": models.vgg13, "VGG16": models.vgg16, "VGG19": models.vgg19,
            "VGG11BN": models.vgg11_bn, "VGG13BN": models.vgg13_bn, "VGG16BN": models.vgg16_bn,
            "VGG19BN": models.vgg19_bn}


class VGGFc(nn.Module):
    def __init__(self, name, hash_bit, increase_scale=True):
        super(VGGFc, self).__init__()
        model_vgg = vgg_dict[name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.hash_layer = nn.Linear(model_vgg.classifier[6].in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.iter_num = 0
        self.__in_features = hash_bit
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale
        self.increase_scale = increase_scale

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        y = self.hash_layer(x)
        if self.increase_scale and self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y

    def output_num(self):
        return self.__in_features


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        std = np.float(
            np.sqrt(6. / (m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * (m.in_channels + m.out_channels))))
        m.weight.data.normal_(0.0, std)
    elif classname.find('Linear') != -1:
        std = np.float(np.sqrt(1. / (m.in_features * m.out_features)))
        m.weight.data.normal_(0.0, std)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class VoxNet(nn.Module):

    def __init__(self, logger, hash_bit, input_shape=(32, 32, 32), n_channels=1):
        super(VoxNet, self).__init__()
        self.body = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv3d(in_channels=n_channels,
                                      out_channels=32, kernel_size=5, stride=2)),
            ('lkrelu1', torch.nn.LeakyReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('lkrelu2', torch.nn.LeakyReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))

        # Trick to accept different input shapes
        x = self.body(torch.autograd.Variable(
            torch.rand((1, 1) + input_shape)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        self.head = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(first_fc_in_features, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
        ]))
        # self.fc = torch.nn.Linear(128, num_classes)
        self.fc = nn.Linear(128, hash_bit)
        # self.fc.weight.data.normal_(0, np.float(np.sqrt(1. / (self.fc.in_features * self.fc.out_features))))
        self.apply(weights_init)
        self.fc.bias.data.fill_(0.0)

    # def feature_parameters(self):
    #     return [self.body.parameters(), self.head.parameters()]
    def get_parameter_list(self):
        voxnet_lr = 0.5
        return [{"params": self.body.parameters(), "lr": voxnet_lr},
                {"params": self.head.parameters(), "lr": voxnet_lr},
                {"params": self.fc.parameters(), "lr": voxnet_lr}]

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        y = self.fc(x)
        output = nn.Tanh()(y)
        return output


class Inception3dLayer(nn.Module):
    def __init__(self, param_dict):
        super(Inception3dLayer, self).__init__()
        self.branch = []
        for i, dictionary in enumerate(param_dict):
            temp_layer_list = []
            for j, layer in enumerate(dictionary['layers']):
                temp_layer_list.append(layer(**dictionary['layer_params'][j]))
                if not (dictionary['activation'][j] is None):
                    temp_layer_list.append(dictionary['activation'][j]())
                if dictionary["bnorm"][j]:
                    if "out_channels" in dictionary["layer_params"][j]:
                        temp_layer_list.append(nn.BatchNorm3d(dictionary["layer_params"][j]["out_channels"]))
                        temp_channels = dictionary["layer_params"][j]["out_channels"]
                    else:
                        temp_layer_list.append(nn.BatchNorm3d(temp_channels))
            self.branch.append(nn.Sequential(*temp_layer_list))
        for i in range(len(self.branch)):
            exec("self.branch" + str(i) + "=self.branch[i]")

    def forward(self, x):
        output = []
        for branch in self.branch:
            output.append(branch(x))
        output = torch.cat(output, dim=1)
        return output


class VRNBasicBlock1(nn.Module):
    def __init__(self, in_channels, drop_rate):
        super(VRNBasicBlock1, self).__init__()
        inception_dict = [{"layers": [nn.Conv3d, nn.Conv3d, nn.Conv3d], \
                           "layer_params": [{"in_channels": in_channels, "out_channels": in_channels / 4, \
                                             "kernel_size": 1, "stride": 1, "padding": 0}, \
                                            {"in_channels": in_channels / 4, "out_channels": in_channels / 4, \
                                             "kernel_size": 3, "stride": 1, "padding": 1}, \
                                            {"in_channels": in_channels / 4, "out_channels": in_channels / 2, \
                                             "kernel_size": 1, "stride": 1, "padding": 0}], \
                           "activation": [nn.ELU, nn.ELU, None], \
                           "bnorm": [True, True, False]}, \
                          {"layers": [nn.Conv3d, nn.Conv3d], \
                           "layer_params": [{"in_channels": in_channels, "out_channels": in_channels / 4, \
                                             "kernel_size": 3, "stride": 1, "padding": 1}, \
                                            {"in_channels": in_channels / 4, "out_channels": in_channels / 2, \
                                             "kernel_size": 3, "stride": 1, "padding": 1}], \
                           "activation": [nn.ELU, None], \
                           "bnorm": [True, False]}
                          ]
        self.bn = nn.BatchNorm3d(in_channels)
        self.activation = nn.ELU()
        self.inception = Inception3dLayer(inception_dict)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x1 = self.drop(self.inception(self.activation(self.bn(x))))
        x = x + x1
        return x


class VRNBasicBlock2(nn.Module):
    def __init__(self, in_channels):
        super(VRNBasicBlock2, self).__init__()
        inception_dict = [
            {"layers": [nn.Conv3d], \
             "layer_params": [{"in_channels": in_channels, "out_channels": in_channels / 2, \
                               "kernel_size": 3, "stride": 2, "padding": 1}], \
             "activation": [None], \
             "bnorm": [True]}, \
            {"layers": [nn.Conv3d], \
             "layer_params": [{"in_channels": in_channels, "out_channels": in_channels / 2, \
                               "kernel_size": 1, "stride": 2, "padding": 0}], \
             "activation": [None], \
             "bnorm": [True]}, \
            {"layers": [nn.Conv3d, nn.MaxPool3d], \
             "layer_params": [{"in_channels": in_channels, "out_channels": in_channels / 2, \
                               "kernel_size": 3, "stride": 1, "padding": 1}, \
                              {"kernel_size": 3, "stride": 2, "padding": 1}], \
             "activation": [None, None], \
             "bnorm": [False, True]}, \
            {"layers": [nn.Conv3d, nn.AvgPool3d], \
             "layer_params": [{"in_channels": in_channels, "out_channels": in_channels / 2, \
                               "kernel_size": 3, "stride": 1, "padding": 1}, \
                              {"kernel_size": 3, "stride": 2, "padding": 1, "count_include_pad": True}], \
             "activation": [None, None], \
             "bnorm": [False, True]}
        ]

        self.bn = nn.BatchNorm3d(in_channels)
        self.activation = nn.ELU()
        self.inception = Inception3dLayer(inception_dict)

    def forward(self, x):
        x = self.inception(self.activation(self.bn(x)))
        return x


class VRN(nn.Module):
    def __init__(self, num_classes, input_shape=(32, 32, 32), n_channels=1):
        super(VRN, self).__init__()
        self.conv0 = nn.Conv3d(n_channels, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

        block1 = VRNBasicBlock1(32, 0.05)
        block2 = VRNBasicBlock1(32, 0.1)
        block3 = VRNBasicBlock1(32, 0.2)
        block4 = VRNBasicBlock2(32)
        self.block_chain1 = nn.Sequential(block1, block2, block3, block4)

        block5 = VRNBasicBlock1(64, 0.3)
        block6 = VRNBasicBlock1(64, 0.4)
        block7 = VRNBasicBlock1(64, 0.5)
        block8 = VRNBasicBlock2(64)
        self.block_chain2 = nn.Sequential(block5, block6, block7, block8)

        block9 = VRNBasicBlock1(128, 0.5)
        block10 = VRNBasicBlock1(128, 0.55)
        block11 = VRNBasicBlock1(128, 0.6)
        block12 = VRNBasicBlock2(128)
        self.block_chain3 = nn.Sequential(block9, block10, block11, block12)

        block13 = VRNBasicBlock1(256, 0.65)
        block14 = VRNBasicBlock1(256, 0.7)
        block15 = VRNBasicBlock1(256, 0.75)
        block16 = VRNBasicBlock2(256)
        self.block_chain4 = nn.Sequential(block13, block14, block15, block16)

        self.conv17 = nn.Conv3d(512, 512, 3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm3d(512)
        self.drop17 = nn.Dropout(0.5)
        self.activation17 = nn.ELU()

        self.bn18 = nn.BatchNorm1d(512)
        self.linear19 = nn.Linear(512, 512)
        self.bn19 = nn.BatchNorm1d(512)
        self.activation19 = nn.ELU()
        self.fc = nn.Linear(512, num_classes)
        self.apply(weights_init)

    def feature_patameters(self):
        return [self.conv0.parameters(), self.block_chain1.parameters(), \
                self.block_chain2.parameters(), self.block_chain3.parameters(), \
                self.block_chain4.parameters(), self.bn17.parameters(), \
                self.conv17.parameters(), self.activation17.parameters(), \
                self.bn18.parameters(), self.linear19.parameters(), \
                self.bn19.paramaters(), self.activation19.parameters()]

    def forward(self, x):
        x = self.conv0(x)

        x = self.block_chain1(x)
        x = self.block_chain2(x)
        x = self.block_chain3(x)
        x = self.block_chain4(x)

        x17 = self.drop17(self.bn17(self.conv17(x)))
        x = x + x17
        x = self.activation17(x)

        x = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)))
        x = x.view(x.size(0), -1)
        x = self.bn18(x)
        x = self.activation19(self.bn19(self.linear19(x)))
        y = self.fc(x)
        return y
