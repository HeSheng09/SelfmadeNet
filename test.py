from model import LeNet_5
import torchvision.models as models


if __name__ == '__main__':
    net=LeNet_5()
    print(net)
    models.vgg11()