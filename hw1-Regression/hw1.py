# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2020/7/14 11:45
"""
import numpy as np
import pandas as pd
import math
import csv

"""
    任务：由前9个小时的18个特征，预测第10个小时的PM2.5（PM2.5是第10个特征）
    训练数据：train.csv是12个月，每个月取20天，每天24小时的数据，每个小时又18个特征  行: 12*20*18 列：24
    测试数据：有18个特征的前9个小时的数据
    参考链接：https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C#scrollTo=dcOrC4Fi-n3i
"""


def load_data():
    # 加载训练数据
    data = pd.read_csv('../data/hw1/train.csv', encoding='big5')
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()  # 4320*24

    # 提取特征，将原始的4320*18，按照每个月分组成为12个18(特征)*480(小时)
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])  # 每月份有：20*24
        for day in range(20):
            sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
        month_data[month] = sample

    # 提取特征，每个月有480个小时，每9个小时的数据成一组，每个月一共有471组数据；所以行为471*12；列是9*18的特征(一个小时的18个特征*9个小时)
    x = np.empty([12 * 471, 18 * 9], dtype=float)  # input
    y = np.empty([12 * 471, 1], dtype=float)  # output
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9) ，reshape(1,-1)转换成一行，转换成一列reshape(-1,1)
                x[month * 471 + day * 24 + hour, :] = month_data[month][:,
                                                      day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                                    -1)
                y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value

    # 归一化处理 axis=0:压缩行，对列求均值
    mean_x = np.mean(x, axis=0)  # 18 * 9
    std_x = np.std(x, axis=0)  # 18 * 9
    for i in range(len(x)):  # 12 * 471
        for j in range(len(x[0])):  # 18 * 9
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    return x, y, mean_x, std_x


# # 分割训练集、验证集
# x_train_set = x[: math.floor(len(x) * 0.8), :]
# y_train_set = y[: math.floor(len(y) * 0.8), :]
# x_validation = x[math.floor(len(x) * 0.8):, :]
# y_validation = y[math.floor(len(y) * 0.8):, :]


# learning_rate = 100
# iter_time = 1000
# eps = 0.0000000001
def train_model(x, y, learning_rate=100, iter_time=1000, eps=0.0000000001):
    """
    :param x: 训练集
    :param y: 标签数据
    :param learning_rate: 学习率
    :param iter_time: 最大迭代次数
    :param eps:为了避免Adagrad分母为0的极小值
    :return:
    """
    # 开始训练数据
    dim = 18 * 9 + 1
    w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float) # np.concatenate追加列

    adagrad = np.zeros([dim, 1])

    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # rmse
        if t % 100 == 0:
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1 x.transpose()表示转置
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight.npy', w)


def test_model(mean_x, std_x):
    testdata = pd.read_csv('../data/hw1/test.csv', header=None, encoding='big5')
    test_data = testdata.iloc[:, 2:].copy()
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()
    test_x = np.empty([240, 18 * 9], dtype=float)
    for i in range(240):
        test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
    w = np.load('weight.npy')
    ans_y = np.dot(test_x, w)

    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)


if __name__ == '__main__':
    x, y, mean_x, std_x = load_data()
    train_model(x, y)
    test_model(mean_x, std_x)
