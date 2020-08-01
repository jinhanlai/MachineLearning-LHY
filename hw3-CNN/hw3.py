import numpy as np
import torch
import pandas as pd
import os
import cv2
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time


def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回传 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))  # 默认读取彩色图片,返回的是numpy.ndarrya
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


"""
    load data
"""
workspace_dir = "../data/hw3/food-11"
print("Loading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

# training 時做 data augmentation（增强）
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
    transforms.RandomRotation(15),  # 随机旋转图片
    transforms.ToTensor(),  # 将图片转成Tensor，并把数值 normalize 到 [0,1] (data normalization)
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

"""
    利用pytorch的torch.utils.data 的 Dataset 及 DataLoader 來"包裝" data，使得后续的train、testing更方便
    Dataset 需要 overload 兩個函數：__len__ 及 __getitem__
    __len__ 必須要回傳 dataset 的大小，而 __getitem__ 則定義了當程式利用 [ ] 取值時，dataset 應該要怎麼回传数据
"""


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y  # label is required to be a LongTensor
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # DataLoader 里面传入的是DataSet对象
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

"""
    Model
"""


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 纬度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128] # 参数：通道数，输出的通道数，卷积核大小(3,意思数3*3)，步长，填充
            nn.BatchNorm2d(64),  # 进行Relu之前对数据进行归一化处理，避免数据过大，出现性能不稳定问题；参数表示特征数量
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64] #参数：卷积核大小，步长、填充

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)  # view 相当于reshape，-1会自动算出纬度是多少
        return self.fc(out)


"""
    Train:使用训练集训练，验证集寻找最好的参数
"""
# model = Classifier().cpu() #mac电脑不支持cpu的gpu
model = Classifier().cpu()  # mac电脑不支持cpu的gpu
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer 使用 Adam，optim必须传入model的参数
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader): # i表示第几个分支，data表示分支对应的数据（list对象，第一个是数据，第二个是标签）
        optimizer.zero_grad()  # 用 optimizer 将 model 參數的 gradient 归零
        # train_pred = model(data[0].cpu())  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        train_pred = model(data[0].cpu())
        batch_loss = loss(train_pred, data[1].cpu())  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item() #获取tesor对应的值

    model.eval()
    with torch.no_grad():  # 下面部分不会计算梯度
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cpu())
            batch_loss = loss(val_pred, data[1].cpu())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

"""
    使用train和validation共同训练
"""
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cpu()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cpu())
        batch_loss = loss(train_pred, data[1].cpu())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        # 將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
          (epoch + 1, num_epoch, time.time() - epoch_start_time, \
           train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))

"""
    Testing
"""
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cpu())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# 將結果寫入 csv 檔
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
