import torch
import torch.nn as nn
from torchstat import stat
from models.converter import UnetGenerator, converter_2, converter_4, converter_6, converter_8
from models.my_models import resnet_18, resnet_34, resnet_152, resnet_101, vgg_13, vgg_16, mobilenet_v2, shufflenet_v2, wideresnet, ShuffleNetG3, vgg_11, vgg_19, SENet18, EfficientNetB0, PNASNetB
from models.models_new import resnet152, vgg19_bn, googlenet, shufflenetv2, inceptionv4

model = converter_8()
stat(model, (3, 32, 32))

exit()