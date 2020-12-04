import argparse
from model import LeNet_5, SelfmadeNet
import torch
from torchvision import datasets,transforms
import sys
import os
import time


def parse_args():
    """
      Parse input arguments
      """
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--model',dest="model",
                        help="model used",
                        default="LeNet5", type=str)
    parser.add_argument('--num_classes', dest='num_classes',
                        help='num_classes',
                        default=10, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs',
                        help='num_epochs',
                        default=2, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='batch_size',
                        default=8, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=2, type=int)
    parser.add_argument('--lr', dest='lr',
                        help='learning rate',
                        default=0.0002, type=float)
    parser.add_argument('--pretrained', dest='pretrained',
                        help="which pretrained model to be load",
                        default='none', type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help="dataset used",
                        default='cifar10', type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help="whether use cuda",
                        default=True, type=bool)
    parser.add_argument('--other', dest='other',
                        help="other information",
                        default="", type=str)
    return parser.parse_args()


def load_pretrained_model(model, pretrained):
    """
    加载预训练模型

    :param model: net model
    :param pretrained: /path/to/pretrained_model
    :return: net model
    """
    if pretrained == "none":
        pass
    else:
        pretrained_path = os.path.abspath(os.path.join(os.getcwd(), pretrained))
        print(pretrained_path)
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path)
            model.load_state_dict(state_dict)
        else:
            print("[error] wrong pretrained model")
            sys.exit(1)
    return model


if __name__ == '__main__':
    data_root = os.path.abspath(os.path.join(os.getcwd(), "data"))

    # 加载参数
    args = parse_args()
    print('Called with args:', end="\t")
    print(args)

    # 是否使用cuda加速
    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("device used for training: {}.".format(device))

    # 加载模型
    if args.model == "LeNet5":
        net = LeNet_5()
        # 加载预训练模型
        net = load_pretrained_model(net, args.pretrained)
        # 修改模型输出
        net.classifier[2] = torch.nn.Linear(84, args.num_classes)
    elif args.model == "SelfmadeNet":
        net = SelfmadeNet()
        # 加载预训练模型
        net = load_pretrained_model(net, args.pretrained)
        # 修改模型输出
        net.classifier[3] = torch.nn.Linear(96, args.num_classes)
    else:
        print("[error] unknown model")
        sys.exit(1)
    net.to(device)
    print(net)

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # 加载训练和验证数据
    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=data_transform['train'])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
        val_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=data_transform['val'])
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.num_workers)
    elif args.dataset == "cifar100":
        train_set = datasets.CIFAR100(root=data_root, train=True, download=True, transform=data_transform['train'])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
        val_set = datasets.CIFAR100(root=data_root, train=False, download=True, transform=data_transform['val'])
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.num_workers)
    else:
        print("[error] invalid dataset")
        sys.exit(1)

    # 损失函数和优化
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # 训练
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^5.2f}%[{}->{}]{:.3f}".format(rate * 100, a, b, loss), end="")
        print()
        print("[epoch {}] time cost on training: {} s.".format(epoch + 1, time.perf_counter() - t1))

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / (len(val_loader) * args.batch_size)
            if val_accurate > best_acc:
                best_acc = val_accurate
                if args.other != "":
                    save_path = os.path.abspath(os.path.join(os.getcwd(), "checkpoints", "{}_{}_{}_{:0>3d}.pth".format(args.model, args.other, args.dataset, int(best_acc*1000))))
                else:
                    save_path = os.path.abspath(os.path.join(os.getcwd(), "checkpoints", "{}_{}_{:0>3d}.pth".format(args.model, args.dataset, int(best_acc*1000))))
                torch.save(net.state_dict(), save_path)
                print(
                    'train_loss: {:.3f}  test_accuracy: {:.3f} \ncheckpoint saved at: {}\n'.format(running_loss / step,
                                                                                                   val_accurate,
                                                                                                   save_path))
            else:
                print(
                    'train_loss: {:.3f}  test_accuracy: {:.3f} \ncheckpoint saved at: {}\n'.format(running_loss / step,
                                                                                                   val_accurate,
                                                                                                   "none"))

    print('Finished Training')