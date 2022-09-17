import torch
import torch.nn as nn


class RainbowModule300(nn.Module):

    def __init__(self):
        super(RainbowModule300, self).__init__()

        self.layers = []

        # for feature map 38x38
        temp_layers = []
        # 19x19
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 10x10
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2,
                                                     padding=1),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 5x5
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 3x3
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2,
                                                     padding=1),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 19x19
        temp_layers = []
        # 38x38
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=1024,
                                                           out_channels=1024,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        # 10x10
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2,
                                                     padding=1),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        # 5x5
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        # 3x3
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2,
                                                     padding=1),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 10x10
        temp_layers = []
        # 19x19
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,
                                                           out_channels=512,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 38x38
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,
                                                           out_channels=512,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 5x5
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 3x3
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2,
                                                     padding=1),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 5x5
        temp_layers = []
        # 10x10
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 19x19
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 38x38
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 3x3
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2,
                                                     padding=1),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 3x3
        temp_layers = []
        # 5x5
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 10x10
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 19x19
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 38x38
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 1x1
        temp_layers = []
        # 3x3
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=3,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 5x5
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 10x10
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 19x19
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 38x38
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        self.layers = nn.ModuleList(self.layers)

    def init_weights(self):
        for module in self.layers.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = []

        for i, block in enumerate(self.layers):
            features += [[]]
            save_feature = x[i].clone()
            for j, layer in enumerate(block):
                if j == i:
                    save_feature = x[i].clone()
                save_feature = layer(save_feature)
                features[i] += [save_feature]

        concat_x = []

        # 38x38
        concat_x += [torch.cat([x[0],
                                features[1][0],
                                features[2][1],
                                features[3][2],
                                features[4][3],
                                features[5][4]], 1)]

        # 19x19
        concat_x += [torch.cat([features[0][0],
                                x[1],
                                features[2][0],
                                features[3][1],
                                features[4][2],
                                features[5][3]], 1)]

        # 10x10
        concat_x += [torch.cat([features[0][1],
                                features[1][1],
                                x[2],
                                features[3][0],
                                features[4][1],
                                features[5][2]], 1)]

        # 5x5
        concat_x += [torch.cat([features[0][2],
                                features[1][2],
                                features[2][2],
                                x[3],
                                features[4][0],
                                features[5][1]], 1)]

        # 3x3
        concat_x += [torch.cat([features[0][3],
                                features[1][3],
                                features[2][3],
                                features[3][3],
                                x[4],
                                features[5][0]], 1)]

        # 1x1
        concat_x += [torch.cat([features[0][4],
                                features[1][4],
                                features[2][4],
                                features[3][4],
                                features[4][4],
                                x[5]], 1)]

        return concat_x


class RainbowModule512(nn.Module):

    def __init__(self):
        super(RainbowModule512, self).__init__()

        self.layers = []

        # for feature map 64x64
        temp_layers = []
        # 32x32
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 16x16
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 8x8
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 4x4
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 2x2
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 32x32
        temp_layers = []
        # 64x64
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=1024,
                                                           out_channels=1024,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        # 16x16
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        # 8x8
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        # 4x4
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        # 2x2
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=1024),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 16x16
        temp_layers = []
        # 32x32
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,
                                                           out_channels=512,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 64x64
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,
                                                           out_channels=512,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 8x8
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 4x4
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 2x2
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=512),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 8x8
        temp_layers = []
        # 16x16
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 32x32
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 64x64
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 4x4
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 2x2
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 4x4
        temp_layers = []
        # 8x8
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 16x16
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 32x32
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 64x64
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 2x2
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 2x2
        temp_layers = []
        # 4x4
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 8x8
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 16x16
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 32x32
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 64x64
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 1x1
        temp_layers += [nn.Sequential(*[nn.AvgPool2d(kernel_size=2,
                                                     stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        # for feature map 1x1
        temp_layers = []
        # 2x2
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 4x4
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 8x8
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 16x16
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 32x32
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        # 64x64
        temp_layers += [nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=256,
                                                           kernel_size=2,
                                                           stride=2),
                                        nn.BatchNorm2d(num_features=256),
                                        nn.ReLU(inplace=True)])]
        self.layers += [nn.ModuleList(temp_layers)]

        self.layers = nn.ModuleList(self.layers)

    def init_weights(self):
        for module in self.layers.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = []

        for i, block in enumerate(self.layers):
            features += [[]]
            save_feature = x[i].clone()
            for j, layer in enumerate(block):
                if j == i:
                    save_feature = x[i].clone()
                save_feature = layer(save_feature)
                features[i] += [save_feature]

        concat_x = []

        # 64x64
        concat_x += [torch.cat([x[0],
                                features[1][0],
                                features[2][1],
                                features[3][2],
                                features[4][3],
                                features[5][4],
                                features[6][5]], 1)]

        # 32x32
        concat_x += [torch.cat([features[0][0],
                                x[1],
                                features[2][0],
                                features[3][1],
                                features[4][2],
                                features[5][3],
                                features[6][4]], 1)]

        # 16x16
        concat_x += [torch.cat([features[0][1],
                                features[1][1],
                                x[2],
                                features[3][0],
                                features[4][1],
                                features[5][2],
                                features[6][3]], 1)]

        # 8x8
        concat_x += [torch.cat([features[0][2],
                                features[1][2],
                                features[2][2],
                                x[3],
                                features[4][0],
                                features[5][1],
                                features[6][2]], 1)]

        # 4x4
        concat_x += [torch.cat([features[0][3],
                                features[1][3],
                                features[2][3],
                                features[3][3],
                                x[4],
                                features[5][0],
                                features[6][1]], 1)]

        # 2x2
        concat_x += [torch.cat([features[0][4],
                                features[1][4],
                                features[2][4],
                                features[3][4],
                                features[4][4],
                                x[5],
                                features[6][0]], 1)]

        # 1x1
        concat_x += [torch.cat([features[0][5],
                                features[1][5],
                                features[2][5],
                                features[3][5],
                                features[4][5],
                                features[5][5],
                                x[6]], 1)]

        return concat_x
