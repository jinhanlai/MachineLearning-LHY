import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cude" if torch.cuda.is_available() else "cpu")


class Adverdataset(Dataset):
    def __init__(self, filepath, label, transform):
        self.filepath = filepath  # 图片所在的路径
        self.label = torch.from_numpy(label).long()
        self.transform = transform
        self.filenames = ["{:03d}".format(i) for i in range(200)]

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.filepath, self.filenames[item] + '.png'))
        img = self.transform(img)
        label = self.label[item]
        return img, label

    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200


class Attacker:
    def __init__(self, img_dir, label):
        self.model = models.vgg16(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把图片normalize到0-1之间，mean为0，variance为1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),  # 把灰度范围从0-255变换到0-1之间
            self.normalize
        ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset(img_dir, label, transform)

        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False)

    # FGSM攻击
    def fgsm_attack(self, image, epsilon, data_grad):
        # 找出 gradient 的方向
        sign_data_grad = data_grad.sign()
        # 將圖片加上 gradient 方向乘上 epsilon 的 noise
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image

    def attack(self, epsilon):
        """

        :param epsilon:
        :return: five groups of images of successful attacks  and  attack failure rate
        """
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        wrong, fail, success = 0, 0, 0  # fail表示攻击失败，wrong表示分类错误，success表示攻击成功
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():
                wrong += 1
                continue

            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            # nll_loss 与CrossEntropyLoss的区别是：
            # torch.nn.CrossEntropyLoss(output,b_y)==F.nll_loss(F.log_softmax(output,1),b_y)
            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]

            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(
                        self.mean, device=device).view(3, 1, 1)
                    adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                    data_raw = data_raw * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(self.mean,
                                                                                                             device=device).view(
                        3, 1, 1)
                    data_raw = data_raw.squeeze().detach().cpu().numpy()
                    # 初始预测结果，攻击后的预测结果，初始数据，攻击后的数据
                    adv_examples.append((init_pred.item(), final_pred.item(), data_raw, adv_ex))
        final_acc = (fail / (wrong + success + fail))

        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return adv_examples, final_acc


if __name__ == '__main__':
    # 讀入圖片相對應的 label
    label_df = pd.read_csv("../data/hw6/data/labels.csv")
    label_df = label_df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv("../data/hw6/data/categories.csv")
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    attacker = Attacker('../data/hw6/data/images', label_df)
    # 要嘗試的 epsilon
    epsilons = [0.1, 0.01]

    accuracies, examples = [], []

    # 進行攻擊 並存起正確率和攻擊成功的圖片
    for eps in epsilons:
        ex, acc = attacker.attack(eps)
        accuracies.append(acc)
        examples.append(ex)

    # 展示图片
    cnt = 0
    plt.figure(figsize=(30, 30))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]) * 2, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, orig_img, ex = examples[i][j]
            # plt.title("{} -> {}".format(orig, adv))
            plt.title("original: {}".format(label_name[orig].split(',')[0]))
            # pytorch里面图片是（channel，height，width），在matplotlib里面是（height，width，channel）
            orig_img = np.transpose(orig_img, (1, 2, 0))
            plt.imshow(orig_img)
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]) * 2, cnt) # subplot(nrow,ncol,index)
            plt.title("adversarial: {}".format(label_name[adv].split(',')[0]))
            ex = np.transpose(ex, (1, 2, 0))
            plt.imshow(ex)
    plt.tight_layout()
    plt.show()
