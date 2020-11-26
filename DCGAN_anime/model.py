import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):

    def __init__(self, in_channels_image, out_channels):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # shape : N x in_channels_image x 64 x 64
            nn.Conv2d(in_channels_image, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_channels, out_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels * 2),  # param is number of out_channels (filter nums)
            nn.LeakyReLU(0.2),

            # shape : N x out_channels*2 x 16 x 16
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_channels * 8, 1, kernel_size=4, stride=2, padding=0),
            # shape : N x 1 x 1 x 1
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):

    def __init__(self, in_channels_noise, out_channels, in_channels_image):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # shape : N x in_channels_noise x 1 x 1
            nn.ConvTranspose2d(in_channels_noise, out_channels * 16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(out_channels * 16),
            nn.ReLU(),

            # N x out_channels*16 x 4 x 4
            nn.ConvTranspose2d(out_channels * 16, out_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels * 8, out_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels * 4, out_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels * 2, in_channels_image, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            # N x out_channels x 64 x64

        )
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)