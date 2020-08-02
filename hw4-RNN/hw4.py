import warnings
import torch
from torch import nn
from torch.utils import data
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
from gensim.models import word2vec, Word2Vec

from sklearn.model_selection import train_test_split

from Preprocess import Preprocess

warnings.filterwarnings('ignore')
"""
    文本分类，判断一条语句有没有恶意
"""


def load_training_data(path):
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()  # 返回的是一个list对象，每一行是一个数据
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x


def load_testing_data(path):
    with open(path, 'r') as f:
        readlines = f.readlines()
        x = ["".join(line.strip('\n').split(',')[1:]).strip() for line in readlines[1:]]
        x = [s.split(' ') for s in x]
    return x


def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


def train_word2vec(x):
    # 单词到向量的词嵌入
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model


# 把数据封装成Dataset
class TwitterDataset(data.Dataset):
    """
        Expected data shape like:(data_num, data_len)
        Data can be a list of numpy array or a list of lists
        input data shape : (data_num, seq_len, feature_dim)

        __len__ will return the number of data
    """

    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


# LSTM 模型
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 嵌入层
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))  # 参数分别是词的个数；词嵌入的纬度
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True) #输入输出的第一维是batch_size
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, 1),  # 用作全连接层
                                        nn.Sigmoid())

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters()) # numel()函数：返回数组中元素的个数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()  # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數
    criterion = nn.BCELoss()  # 定義損失函數，這裡我們使用 binary cross entropy loss
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 將模型的參數給 optimizer，並給予適當的 learning rate
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 這段做 training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)  # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            labels = labels.to(device,
                               dtype=torch.float)  # device為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
            optimizer.zero_grad()  # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            outputs = model(inputs)  # 將 input 餵給模型
            outputs = outputs.squeeze()  # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion() eg:[[1],[2]]--> [1,2]
            loss = criterion(outputs, labels)  # 計算此時模型的 training loss
            loss.backward()  # 算 loss 的 gradient
            optimizer.step()  # 更新訓練模型的參數
            correct = evaluation(outputs, labels)  # 計算此時模型的 training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch + 1, i + 1, t_batch, loss.item(), correct * 100 / batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))

        # 這段做 validation
        model.eval()  # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)  # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
                labels = labels.to(device,
                                   dtype=torch.float)  # device 為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
                outputs = model(inputs)  # 將 input 餵給模型
                outputs = outputs.squeeze()  # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
                loss = criterion(outputs, labels)  # 計算此時模型的 validation loss
                correct = evaluation(outputs, labels)  # 計算此時模型的 validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch * 100))
            if total_acc > best_acc:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                # torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc / v_batch * 100))
            print('-----------------------------------------------')
            model.train()  # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數（因為剛剛轉成 eval 模式）


def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1  # 大於等於 0.5 為正面
            outputs[outputs < 0.5] = 0  # 小於 0.5 為負面
            ret_output += outputs.int().tolist()
    return ret_output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_with_label = os.path.join("../data/hw4/training_label.txt")
    train_no_label = os.path.join("../data/hw4/training_nolabel.txt")
    testing_data = os.path.join("../data/hw4/testing_data.txt")

    w2v_path = os.path.join("../hw4-RNN/w2v_all.model")

    # 固定句子长度，要不要固定 embedding ，batch大小，训练epoch、learning rate
    sen_len = 20
    fix_embedding = True
    batch_size = 30
    epoch = 5
    lr = 0.001

    print("loading training data...")
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)

    print("loading testing data...")
    test_x = load_testing_data(testing_data)

    if not os.path.exists("w2v_all.model"):
        print("saving word to vec model...")
        model = train_word2vec(train_x + test_x)
        model.save(os.path.join('./', "w2v_all.model"))

    # 对input 和 labels 做预处理
    preprocess = Preprocess(train_x, sen_len, w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # 製作一個 model 的對象
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5,
                     fix_embedding=fix_embedding)
    model = model.to(device)  # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

    # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
    X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

    # 把 data 做成 dataset 供 dataloader 取用
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    # 把 data 轉成 batch of tensors
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)
    training(batch_size, epoch, lr, '../hw4-RNN', train_loader, val_loader, model, device)

    # 開始測試模型並做預測
    print("loading testing data ...")
    test_x = load_testing_data(testing_data)
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=8)
    print('\nload model ...')
    model = torch.load(os.path.join('../hw4-RNN/', 'ckpt.model'))
    outputs = testing(batch_size, test_loader, model, device)

    # 寫到 csv 檔案供上傳 Kaggle
    tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join('../hw4-RNN/', 'predict.csv'), index=False)
    print("Finish Predicting")
