# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2020/7/17 23:40
"""
import numpy as np
import matplotlib.pyplot as plt

"""
    二分类问题，根据给定的数据判断这个人的年收入是否高于5万美元；分别使用了logistic regression 与 generative model
    参考链接https://colab.research.google.com/drive/1JaMKJU7hvnDoUfZjvUKzm9u-JLeX6B2C#scrollTo=ox7joE3aZkh-
"""
def load_file(path):
    with open(path) as f:
        next(f)
        data = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
    return data


def _normalize(X, train=True, specified_colum=None, X_mean=None, X_std=None):
    """
    :param X: data to be processed
    :param train: 'True' when processing training data, 'False' for testing data
    :param specified_colum: indexes of the columns that will be normalized. If 'None', all columns will be normalized.
    :param X_mean: mean value of training data, used when train = 'False'
    :param X_std: standard deviation of training data, used when train = 'False'
    :return: X: normalized data, X_mean: computed mean value of training data ,
            X_std: computed standard deviation of training data

    """
    if specified_colum is None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std


def _train_dev_split(X, Y, dev_ratio=0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize], Y[randomize]


def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)  # np.matmul矩阵乘法


def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)


def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label.T, np.log(y_pred)) - np.dot((1 - Y_label).T, np.log(1 - y_pred))
    return cross_entropy


def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum((pred_error * X), 0).reshape(-1, 1)  # 转变成一列
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


def load_data():
    np.random.seed(0)
    X_train_fpath = '../data/hw2/data/X_train'
    Y_train_fpath = '../data/hw2/data/Y_train'
    X_test_fpath = '../data/hw2/data/X_test'

    X_train = load_file(X_train_fpath)
    Y_train = load_file(Y_train_fpath)
    X_test = load_file(X_test_fpath)
    # Normalize training and testing data
    X_train, X_mean, X_std = _normalize(X_train, train=True)
    X_test, _, _ = _normalize(X_test, train=False, specified_colum=None, X_mean=X_mean, X_std=X_std)

    return X_train, Y_train, X_test


def train_min_batch(X_train, Y_train, max_iter=10, batch_size=8, learning_rate=0.2):
    X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=0.1)
    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]
    data_dim = X_train.shape[1]

    # Zero initialization for weights ans bias
    w = np.zeros((data_dim, 1))
    b = np.zeros((1,))

    # Keep the loss and accuracy at every iteration for plotting
    train_loss = []
    dev_loss = []
    train_acc = []
    dev_acc = []

    # Calcuate the number of parameter updates
    step = 1

    # Iterative training
    for epoch in range(max_iter):
        # Random shuffle at the begging of each epoch
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Mini-batch training
        for idx in range(int(np.floor(train_size / batch_size))):
            X = X_train[idx * batch_size:(idx + 1) * batch_size]
            Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

            # Compute the gradient
            w_grad, b_grad = _gradient(X, Y, w, b)

            # gradient descent update
            # learning rate decay with time
            w = w - learning_rate / np.sqrt(step) * w_grad
            b = b - learning_rate / np.sqrt(step) * b_grad

            step = step + 1

        # Compute loss and accuracy of training set and development set
        y_train_pred = _f(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(_accuracy(Y_train_pred, Y_train))
        train_loss.append(float(_cross_entropy_loss(y_train_pred, Y_train) / train_size))

        y_dev_pred = _f(X_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
        dev_loss.append(float(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size))

    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))
    return train_loss, dev_loss, train_acc, dev_acc, w, b


def plot_ans(train_loss, dev_loss, train_acc, dev_acc):
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.title('Loss')
    plt.legend(['train', 'dev'])
    plt.savefig('loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    plt.savefig('acc.png')
    plt.show()


def test(X_test, w, b, savename):
    output_fpath = 'output_{}.csv'
    X_test_fpath = '../data/hw2/data/X_test'
    predictions = _predict(X_test, w, b)
    with open(output_fpath.format(savename), 'w') as f:
        f.write('id,label\n')
        for i, label in enumerate(predictions):
            f.write('{},{}\n'.format(i, label))

    # Print out the most significant weights columns
    ind = np.argsort(np.abs(w), 0)[::-1]
    with open(X_test_fpath) as f:
        content = f.readline().strip('\n').split(',')
    features = np.array(content)
    for i in ind[0:10]:
        print(features[i], w[i])


def train_generative_model(X_train, Y_train):
    X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
    X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])
    data_dim = X_train.shape[1]

    mean_0 = np.mean(X_train_0, axis=0)
    mean_1 = np.mean(X_train_1, axis=0)

    # Compute in-class covariance
    cov_0 = np.zeros((data_dim, data_dim))
    cov_1 = np.zeros((data_dim, data_dim))

    for x in X_train_0:
        cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
    for x in X_train_1:
        cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

    # Shared covariance is taken as a weighted average of individual in-class covariance.
    cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

    # Computing weights and bias
    u, s, v = np.linalg.svd(cov, full_matrices=False)  # svd分解的目的是计算con的逆
    inv = np.matmul(v.T * 1 / s, u.T)

    # Directly compute weights and bias
    w = np.dot(inv, mean_0 - mean_1)
    b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) \
        + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

    # Compute accuracy on training set
    # 维度太高，cpu爆炸
    # Y_train_pred = 1 - _predict(X_train, w, b)
    # print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))

    return w, b


if __name__ == '__main__':
    X_train, Y_train, X_test = load_data()
    train_loss, dev_loss, train_acc, dev_acc, w, b = train_min_batch(X_train, Y_train)
    # plot_ans(train_loss, dev_loss, train_acc, dev_acc)
    test(X_test, w, b, "logistic")
    w2, b2 = train_generative_model(X_train, Y_train)
    test(X_test, w2, b2, "generative")
