from model import LeNet_5
import torchvision.models as models
import os


if __name__ == '__main__':
    accs = [0.001, 0.010, 0.647, 0.653]
    for acc in accs:
        print("{:0>3d}".format(int(acc * 1000)))