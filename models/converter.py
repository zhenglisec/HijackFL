import functools
from turtle import forward

import torch
import torch.nn as nn


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator_mnist(nn.Module):
    def __init__(self, input_c=1, output_nc=1, num_downs=7, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Tanh):
        super(UnetGenerator_mnist, self).__init__()
        # construct unet structure
        self.unet_block_real_img = UnetPre(input_c=1)
        self.unet_block_sec_img = UnetPre(input_c=1)

        self.model = nn.Sequential(
            nn.Conv2d(3,ngf, kernel_size=3, stride=1, padding=1,bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            nn.Conv2d(ngf,ngf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            nn.Conv2d(ngf,ngf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            nn.Conv2d(ngf, input_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )
    def forward(self, input, sec_img):
        real_img_feature = self.unet_block_real_img(input)
        sec_img_feature = self.unet_block_sec_img(sec_img)
        contain_img = torch.cat([real_img_feature, sec_img_feature], dim=1)
        output = self.model(contain_img)
        return output

class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=6, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Tanh):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        self.unet_block_real_img = UnetPre(input_c=3)
        # self.unet_block_sec_img = UnetPre(input_c=3)
        
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        #for i in range(num_downs - 5):
            #unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 #norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block
    def forward(self, input):

        real_img_feature = self.unet_block_real_img(input)
        # sec_img_feature = self.unet_block_sec_img(sec_img)
        # contain_img = torch.cat([real_img_feature, sec_img_feature], dim=1)
        output = self.model(real_img_feature)
        return output

class UnetPre(nn.Module):
    def __init__(self, input_c):
        super(UnetPre, self).__init__()
        '''
        self.pre_prepare1 = nn.Sequential(
            nn.Linear(1600, 1400),
            nn.BatchNorm1d(1400),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1400, 1200),
            nn.BatchNorm1d(1200),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1200, 1 * 32 * 32),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        '''

        self.pre = nn.Sequential(
            nn.ConvTranspose2d(input_c, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, input):
        out = self.pre(input)
        return out

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            elif output_function == nn.Sigmoid:
                up = [uprelu, upconv, nn.Sigmoid()]
            else:
                up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)



import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn



class converter_2(nn.Module):
    def __init__(self, input_c=3):
        super(converter_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 96, kernel_size=4, stride=2, padding=0,bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 3, kernel_size=4, stride=2, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(48),
            # nn.LeakyReLU()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class converter_4(nn.Module):
    def __init__(self, input_c=3):
        super(converter_4, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 48, kernel_size=3, stride=2, padding=1,bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96,48, kernel_size=4, stride=2, padding=1,bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(48),
            # nn.LeakyReLU()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class converter_6(nn.Module):
    def __init__(self, input_c=3):
        super(converter_6, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 24, kernel_size=3, stride=2, padding=1,bias=True),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24,48, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96,48, kernel_size=4, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(48,24, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(48),
            # nn.LeakyReLU()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class converter_8(nn.Module):
    def __init__(self, input_c=3):
        super(converter_8, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 12, kernel_size=3, stride=2, padding=1,bias=True),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            nn.Conv2d(12,24, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24,48, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96,48, kernel_size=4, stride=2, padding=0,bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(48,24, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24,12, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=3, stride=1, padding=0, bias=True),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class converter_10(nn.Module):
    def __init__(self, input_c=3):
        super(converter_10, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 12, kernel_size=3, stride=2, padding=1,bias=True),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            nn.Conv2d(12,24, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24,48, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 128, kernel_size=2, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,96, kernel_size=1, stride=2, padding=0,bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(96,48, kernel_size=3, stride=2, padding=0,bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(48,24, kernel_size=3, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24,12, kernel_size=3, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=2, stride=1, padding=0, bias=True),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x

def train_converter(Transformer, target_model, dataloder, hlpr):
    # Transformer = Transformer.to(hlpr.params.device)
    target_model.eval()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(Transformer.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    for param in target_model.parameters():
        param.requires_grad = False

    # if hlpr.params.model_arch == 'vgg_16':
    #     learning_rate = 5  # 5 best for vgg_16  0.5 best for resnet  10 for mobilenet_v2, 0.1 for wideresnet
    # elif hlpr.params.model_arch == 'resnet_34':
    #     learning_rate = 0.5
    # elif hlpr.params.model_arch == 'mobilenet_v2':
    #     learning_rate = 10
    # elif hlpr.params.model_arch == 'wideresnet':
    #     learning_rate = 0.1

    learning_rate = 0.1
    Transformer_optimizer = optim.Adam(Transformer.parameters(), lr=learning_rate)
    
    first_drop = int(hlpr.params.converter_epochs * 0.5)
    second_drop = int(hlpr.params.converter_epochs * 0.8)
    milestones = [first_drop, second_drop]
    Transformer_scheduler = torch.optim.lr_scheduler.MultiStepLR(Transformer_optimizer, milestones, gamma=0.1, last_epoch=-1)

    
    for epoch in range(0, hlpr.params.converter_epochs):
        train_loss = 0
        train_correct = 0
        train_total = 0

        ######  training Transformer  ######
        Transformer.train()
        for batch_idx, (inputs, targets) in enumerate(dataloder):
            inputs, targets = inputs.to(hlpr.params.device), targets.to(hlpr.params.device)
            
            inputs_T = Transformer(inputs)
            outputs = target_model(inputs_T)
            outputs = outputs[:, :10]
            loss = criterion(outputs, targets)

            Transformer_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            Transformer_optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        epoch_acc = train_correct/train_total
        epoch_loss = train_loss/(batch_idx+1)
        print('Training', epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*epoch_acc, train_correct, train_total))
        ######  testing Transformer  ######
        Transformer.eval()
        lowest_loss = 1000
        test_total = 0
        test_correct = 0
        test_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloder):
            inputs, targets = inputs.to(hlpr.params.device), targets.to(hlpr.params.device)
            
            inputs_T = Transformer(inputs)
            outputs = target_model(inputs_T)
            outputs = outputs[:, :10]
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

        epoch_acc = test_correct/test_total
        epoch_loss = test_loss/(batch_idx+1)
        print('Testing', epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*epoch_acc, test_correct, test_total))
        Transformer_scheduler.step()
        if epoch_loss <= lowest_loss:
            lowest_loss = epoch_loss
            print("Transformer save successfully.")
            torch.save({'model_state_dict': Transformer.state_dict()}, '{0}/transformer.pth'.format(hlpr.params.folder_path))

    for param in target_model.parameters():
        param.requires_grad = True

    checkpoint = torch.load('{0}/transformer.pth'.format(hlpr.params.folder_path))
    print("Transformer load successfully.")
    Transformer.load_state_dict(checkpoint['model_state_dict'])
    Transformer.eval()