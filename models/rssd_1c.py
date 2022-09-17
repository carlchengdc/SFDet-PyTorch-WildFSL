import os
import torch
import torch.nn as nn
from layers.detection import Detect
from torchvision.models import vgg16_bn
from layers.rainbow_module import RainbowModule300, RainbowModule512


class RSSD_1C(nn.Module):

    """
    RSSD_1C architecture
    """

    def __init__(self,
                 mode,
                 base,
                 extras,
                 rainbow_layers,
                 head,
                 anchors,
                 class_count):
        super(RSSD_1C, self).__init__()

        self.mode = mode
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.rainbow_layers = rainbow_layers
        self.class_head = head[0]
        self.loc_head = head[1]
        self.anchors = anchors
        self.class_count = class_count

        if mode == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect.apply

    def forward(self, x):
        sources = []
        class_preds = []
        loc_preds = []

        b, _, _, _ = x.shape
        # apply vgg up to conv4_3 relu
        for i in range(32):
            x = self.base[i](x)
        sources.append(x)

        # apply vgg up to fc7
        for i in range(32, len(self.base)):
            x = self.base[i](x)
        sources.append(x)

        # apply extras
        for i, layer in enumerate(self.extras):
            x = layer(x)
            if i % 2 == 1:
                sources.append(x)

        concat_x = self.rainbow_layers(sources)

        # apply multibox head to sources
        class_head = self.class_head
        loc_head = self.loc_head
        for x in concat_x:
            class_preds.append(class_head(x).permute(0, 2, 3, 1).contiguous())
            loc_preds.append(loc_head(x).permute(0, 2, 3, 1).contiguous())

        class_preds = torch.cat([pred.view(b, -1) for pred in class_preds], 1)
        loc_preds = torch.cat([pred.view(b, -1) for pred in loc_preds], 1)

        class_preds = class_preds.view(b, -1, self.class_count)
        loc_preds = loc_preds.view(b, -1, 4)

        if self.mode == 'test':
            output = self.detect(self.class_count,
                                 self.softmax(class_preds),
                                 loc_preds,
                                 self.anchors)
        else:
            output = (class_preds,
                      loc_preds)
        return output

    def init_weights(self, model_save_path, basenet):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def load_weights(self,
                     base_file):

        other, ext = os.path.splitext(base_file)

        if ext == '.pkl' or ext == '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage,
                                            loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def get_extras(config, in_channels, batch_norm=False):
    layers = []
    flag = False

    for i, out_channels in enumerate(config):
        if in_channels != 'S':
            if out_channels == 'S':
                layer = [nn.Conv2d(in_channels=in_channels,
                                   out_channels=config[i + 1],
                                   kernel_size=(1, 3)[flag],
                                   stride=2,
                                   padding=1),
                         nn.BatchNorm2d(num_features=config[i + 1]),
                         nn.ReLU(inplace=True)]
                layers += [nn.Sequential(*layer)]
            else:
                layer = [nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=(1, 3)[flag]),
                         nn.BatchNorm2d(num_features=out_channels),
                         nn.ReLU(inplace=True)]
                layers += [nn.Sequential(*layer)]
            flag = not flag
        in_channels = out_channels

    return layers


def multibox(class_count,
             new_size):

    if new_size == 300:
        in_channels = 2816
    elif new_size == 512:
        in_channels = 3072

    class_layer = nn.Conv2d(in_channels=in_channels,
                            out_channels=6 * class_count,
                            kernel_size=3,
                            padding=1)
    loc_layer = nn.Conv2d(in_channels=in_channels,
                          out_channels=6 * 4,
                          kernel_size=3,
                          padding=1)

    return class_layer, loc_layer


extras_config = {
    '300': [256, 'S',
            512, 128, 'S',
            256, 128, 256, 128, 256],
    '512': [256, 'S',
            512, 128, 'S',
            256, 128, 'S',
            256, 128, 'S',
            256, 128, 'S',
            256]
}


def build_RSSD_1C(mode,
                  new_size,
                  anchors,
                  class_count):

    base = vgg16_bn(weights='IMAGENET1K_V1')
    base = [x for x in base.features]
    base[23] = nn.MaxPool2d(kernel_size=2,
                            stride=2,
                            ceil_mode=True)
    base[43] = nn.MaxPool2d(kernel_size=3,
                            stride=1,
                            padding=1)
    base += [nn.Conv2d(in_channels=512,
                       out_channels=1024,
                       kernel_size=3,
                       stride=1,
                       padding=6,
                       dilation=6,
                       )]
    base += [nn.BatchNorm2d(num_features=1024)]
    base += [nn.ReLU(inplace=True)]
    base += [nn.Conv2d(in_channels=1024,
                       out_channels=1024,
                       kernel_size=1,
                       stride=1)]
    base += [nn.BatchNorm2d(num_features=1024)]
    base += [nn.ReLU(inplace=True)]

    extras = get_extras(config=extras_config[str(new_size)],
                        in_channels=1024)

    if new_size == 300:
        rainbow_layers = RainbowModule300()
    elif new_size == 512:
        rainbow_layers = RainbowModule512()

    head = multibox(class_count=class_count,
                    new_size=new_size)

    return RSSD_1C(mode,
                   base,
                   extras,
                   rainbow_layers,
                   head,
                   anchors,
                   class_count)
