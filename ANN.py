import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import sys
import gzip


def load_mnist(path, kind='train'):

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:

        magic, n = struct.unpack('>II', lbpath.read(8))

        labels = np.fromstring(lbpath.read(), dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromstring(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def _test_show(X_train, y_train):  # 按照2*5的方式排列显示单个数字的图像
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)  # 两行五列

    ax = ax.flatten()  # 折叠成一维数组
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def _test_show7(X_train, y_train):  # 绘制数字7的前25个不同变体
    fig, ax = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, )
    ax = ax.flatten()
    for i in range(25):
        img = X_train[y_train == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


class HandWritingRecognition(object):  # 初始化
    def __init__(self, n_output, n_features, n_hidden, epochs=10, eta=0.001,
                 shuffle=True, minibatches=1):

        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):  # 对结果进行onehot编码

        onehot = np.zeros((k, y.shape[0]))
        for idx, val, in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):  # 权重初始化
        # 计算权重

        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):  # 激活函数
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_gradient(self, z):  # 激活函数导数 for 反向传播
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):  # 增加偏置，最后一项增加b
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X

        return X_new

    def _feedforward(self, X, w1, w2):  # 前向传播进行计算
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _get_cost(self, y_enc, output):
        cost = np.sum((y_enc - output)**2)/2
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w2, w1):  # 反向传播
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        return grad1, grad2

    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)  # 找到最大的（最有可能是的数字）
        # print(y_pred)
        return y_pred

    def fit(self, X, y, print_progress):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        for i in range(self.epochs):

            if print_progress:  # 输出运行进度
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])  # 随机排序
                X_data, y_data = X_data[idx], y_data[idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # 前馈
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx], output=a3)
                self.cost_.append(cost)

                # 通过反向传播计算梯度
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:, idx],
                                                  w1=self.w1, w2=self.w2)

                # 更新权重
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= delta_w1
                self.w2 -= delta_w2

        return self


def costplt(nn, minibatches):

    batches = np.array_split(range(len(nn.cost_)), minibatches)
    cost_array = np.array(nn.cost_)
    cost_ave = [np.mean(cost_array[i]) for i in batches]

    plt.plot(range(len(cost_ave)), cost_ave, color='red')
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = r'C:\Users\llxpo\Desktop\大二下\最优化\实验\mnist'  # 路径

    X_train, y_train = load_mnist(path, kind='train')  # X_train : 60000*784
    X_test, y_test = load_mnist(path, kind='t10k')  # X_test : 10000*784
    # 得到训练样本和测试数据

    # 测试一下输出图像查看是否正确
    # _test_show(X_train=X_train, y_train=y_train)
    # _test_show7(X_train=X_train, y_train=y_train)
    minibatches = 20
    epochs = 150
    # 初始化对象（模型）
    nn = HandWritingRecognition(n_output=10, n_features=X_train.shape[1], n_hidden=120,
                                epochs=epochs, eta=0.001, shuffle=True, minibatches=minibatches)

    nn.fit(X_train, y_train, print_progress=True)

    # 绘图（误差曲线）
    costplt(nn, minibatches * epochs)

    print()
    y_train_pred = nn.predict(X_train)
    # print(y_train_pred)
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('训练准确率:', end="")
    print(acc * 100, end="%\n\n")

    y_test_pred = nn.predict(X_test)
    # print(y_test_pred)
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('测试准确率:', end="")
    print(acc * 100, end="%")
