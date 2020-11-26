import argparse

class Train_arguments():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--data_root', default='D:\coding\Projects\datasets/faces',type=str, help='file root')

        self.parser.add_argument('--reshape_size', default=64, type=int, help='reshape size of image')

        self.parser.add_argument('--batch_size', default=64, type=int, help='batch size of input')
        self.parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
        self.parser.add_argument('--img_channel', default=3, type=int, help='image channel')
        self.parser.add_argument('--noise_channel', default=256, type=int, help='noise_channel')
        self.parser.add_argument('--epochs', type=int, default=20, help='training epochs')
        self.parser.add_argument('--out_channels', default=64, type=int, help='num of filters for first Conv layer'
                                                                            ', decide the model complexity')


    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()
        self.args = args

        return self.args

