import matplotlib.pyplot as plt
import math


def read_all(file):
    a, b, c = [], [], []
    step = 0
    k = 1
    with open(file, 'r') as f:
        lines = f.readlines()  # 读取文件内容并存到列表中
        for i in lines:
            value = [s for s in i.split()]  # 这里s和value都是去掉了\n
            a.append(value)
            if len(a) % k == 0:
                x = [float(z[0]) for z in a]  # 此处x为长度为k的list
                x = sum(x) / k
                b.append(x)  # 纵轴
                c.append(k + step * k)  # 横轴
                step += 1
                a = []
    return b, c


def draw(file, label, ave_step=1, linewidth=1, color='red'):
    a, b, c = [], [], []
    step = 0
    k = ave_step
    with open(file, 'r') as f:
        lines = f.readlines()  # 读取文件内容并存到列表中
        for i in lines:
            value = [s for s in i.split()]  # 这里s和value都是去掉了\n
            a.append(value)
            if len(a) % k == 0:
                x = [float(z[0]) for z in a]  # 此处x为长度为k的list
                x = sum(x) / k
                b.append(x)  # 纵轴
                c.append(k + step * k)  # 横轴
                step += 1
                a = []
    plt.plot(c, b, label=label, linewidth=linewidth, color=color)


def colordraw(file, label, ave_step, linewidth=1):
    a, b, c = [], [], []
    step = 0
    k = ave_step
    with open(file, 'r') as f:
        lines = f.readlines()  # 读取文件内容并存到列表中
        for i in lines:
            value = [s for s in i.split()]  # 这里s和value都是去掉了\n
            a.append(value)
            if len(a) % k == 0:
                x = [float(z[0]) for z in a]  # 此处x为长度为k的list
                x = sum(x) / k
                b.append(x)  # 纵轴
                # c.append(k + step * k)  # 横轴
                c.append(1 + step)
                step += 1
                a = []
    plt.plot(c, b, label=label, linewidth=linewidth)


def log_colordraw(file, label, ave_step, linewidth=1):
    a, b, c = [], [], []
    step = 0
    k = ave_step
    with open(file, 'r') as f:
        lines = f.readlines()  # 读取文件内容并存到列表中
        for i in lines:
            value = [s for s in i.split()]  # 这里s和value都是去掉了\n
            a.append(value)
            if len(a) % k == 0:
                x = [float(z[0]) for z in a]  # 此处x为长度为k的list
                x = sum(x) / k
                b.append(math.log10(x))  # 纵轴
                # c.append(k + step * k)  # 横轴
                c.append(1 + step)
                step += 1
                a = []
    plt.plot(c, b, label=label, linewidth=linewidth)


def picture_return_time():
    plt.figure()
    ave_step = 1
    colordraw('SAC_0.01alpha_return.txt', 'return', ave_step)
    colordraw('SAC_noTemp_0.001alpha_return.txt', 'return_no', ave_step)
    colordraw('SAC_noTemp_0.01alpha_return.txt', 'return_no_high', ave_step)
    colordraw('SAC_noTemp_0.0001alpha_return.txt', 'return_no_low', ave_step)
    # colordraw('return_DQN.txt', 'return_DQN', 40)
    plt.legend()

    plt.figure()
    colordraw('SAC_0.01alpha_time.txt', 'time', ave_step)
    colordraw('SAC_noTemp_0.001alpha_time.txt', 'time_no', ave_step)
    colordraw('SAC_noTemp_0.01alpha_time.txt', 'time_no_high', ave_step)
    colordraw('SAC_noTemp_0.0001alpha_time.txt', 'time_no_low', ave_step)
    # colordraw('time_DQN.txt', 'time_DQN', 40)
    plt.legend()


def picture_h():
    ave_step = 1
    plt.figure()
    colordraw('entropy.txt', 'H', ave_step)
    colordraw('entropy_noTemp_0.001alpha.txt', 'H_no', ave_step)
    colordraw('entropy_noTemp_0.01alpha.txt', 'H_no_high', ave_step)
    colordraw('entropy_noTemp_0.0001alpha.txt', 'H_no_low', ave_step)
    plt.legend()


def picture_seed():
    ave_step = 1
    plt.figure()
    colordraw('SAC_0.01alpha_seed1_time.txt', 'time-SAC-seed1', ave_step)
    colordraw('SAC_0.01alpha_seed2_time.txt', 'time-SAC-seed2', ave_step)
    colordraw('SAC_0.01alpha_seed3_time.txt', 'time-SAC-seed3', ave_step)
    colordraw('SAC_0.01alpha_seed4_time.txt', 'time-SAC-seed4', ave_step)
    plt.legend()

    plt.figure()
    colordraw('SAC_0.01alpha_seed1_return.txt', 'return-SAC-seed1', ave_step)
    colordraw('SAC_0.01alpha_seed2_return.txt', 'return-SAC-seed2', ave_step)
    colordraw('SAC_0.01alpha_seed3_return.txt', 'return-SAC-seed3', ave_step)
    colordraw('SAC_0.01alpha_seed4_return.txt', 'return-SAC-seed4', ave_step)
    plt.legend()

    plt.figure()
    colordraw('entropy_seed1.txt', 'entropy-SAC-seed1', ave_step)
    colordraw('entropy_seed2.txt', 'entropy-SAC-seed2', ave_step)
    colordraw('entropy_seed3.txt', 'entropy-SAC-seed3', ave_step)
    colordraw('entropy_seed4.txt', 'entropy-SAC-seed4', ave_step)
    plt.legend()


def main():
    # picture_h()
    # picture_return_time()
    picture_seed()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()