import torch
import torch.nn as nn
import brevitas.onnx as bo
import brevitas.nn as qnn



configuration = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = qnn.QuantConv2d(in_channels = in_channels, out_channels = out_channels, return_quant_tensor=True,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.ReLU() #self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):

    def __init__(self, in_channels = 3, **kwargs):
        super(Yolov1, self).__init__()
        self.configuration = configuration
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.configuration)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        # return x
        # return self.fcs(torch.flatten(x,start_dim = 1))
        return self.fcs(x)
    def _create_conv_layers(self, configuration):
        layers = []
        in_channels = self.in_channels

        for x in configuration:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                     in_channels, x[1], kernel_size = x[0], stride = x[2], padding = x[3],
                    )
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size = conv1[0],
                            stride = conv1[2],
                            padding = conv1[3],
                            )
                    ]
                    layers += [
                        CNNBlock(conv1[1], 
                        conv2[1], 
                        kernel_size = conv2[0], 
                        stride = conv2[2], 
                        padding = conv2[3],
                        )
                    ]

                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(1024 * S * S, 496, bias = False),
            nn.ReLU(),
            qnn.QuantLinear(496, S * S * (C + B * 5), bias = False),
        )

def test(S = 7, B = 2, C = 1):
    model = Yolov1(split_size = S, num_boxes = B, num_classes = C)
    print(model)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)
    print("Transforming...")
    bo.export_finn_onnx(model, (1, 3, 448, 448), "YOLOv1.onnx")
    print("Success!!")

# test()