import matplotlib.pyplot as plt
import os
import sys
import re


def read_log(log_path, headlines=5, gap=4):
    if not os.path.exists(log_path):
        print("log doesn't exist!")
        sys.exit(1)
    else:
        ts, losses, accs, pths = [], [], [], []
        with open(log_path, "r") as f:
            lines = f.readlines()
            for i in range(int((len(lines) - headlines-1) / gap)):
                # re.findall(r"a(.+?)b", str)
                t = float(re.findall(r"[0-9]+?\].+?(\d+(\.\d+)?)", lines[headlines+1 + i * gap])[0][0])
                loss, loss_de, acc, acc_de = \
                re.findall(r"[a-z_]+?: (\d+(\.\d+)?)\s+?[a-z_]+?: (\d+(\.\d+)?)", lines[headlines+2 + i * gap])[0]
                # acc=float(re.findall(r"[a-z_]+?: (\d+(\.\d+)?)",lines[6+i*4].split("  ")[1])[0][0])
                pth = lines[headlines+3 + i * gap].split(": ")[-1].split("\n")[0]
                ts.append(t)
                losses.append(float(loss))
                accs.append(float(acc))
                pths.append(pth)
        return ts,losses,accs,pths


if __name__ == '__main__':
    ts,losses,accs,pths = read_log("./results/LeNet5-cifar10.out",headlines=20,gap=5)
    # time consuming
    x=[i+1 for i in range(len(ts))]
    plt.plot(x, ts)
    avg_time=sum(ts)/len(ts)
    plt.axhline(y=avg_time,c="red",ls=":")
    plt.text(-2,avg_time,"{:.2f}".format(avg_time))
    plt.title("time consuming on each epoch")
    plt.xlabel("epoch")
    plt.ylabel("time(s)")
    plt.show()
    # training loss
    plt.plot(x, losses)
    best_loss=min(losses)
    plt.axhline(y=best_loss,c="red",ls=":")
    plt.text(-7,best_loss-0.008,best_loss)
    plt.title("training loss of each epoch")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.show()
    # test accuracy
    plt.plot(x, accs)
    best_acc=max(accs)
    plt.axhline(y=best_acc, c="red", ls=":")
    plt.text(-6,best_acc-0.004,best_acc)
    plt.title("test accuracy of each epoch")
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.show()
