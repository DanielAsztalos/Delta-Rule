import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

def plot_ims(imgs, labels):
    fig = plt.figure(figsize=(6, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(6, 4), axes_pad=0.7)

    for ax, im in zip(grid, imgs[:24]):
        ax.imshow(im[0], cmap='gray')
        ax.set_title(labels[im[1]])

def read_data():
    data_all = []
    data = os.listdir("data")
    labels = []
    for i, direct in enumerate(data):
        imgs = os.listdir("data/" + direct)
        for img in imgs:
            x = Image.open("data/" + direct + "/" + img).convert('L').resize((64, 64))
            data_all.append((np.array(x), i))
        labels.append(direct)
    return data_all, labels

def preprocess_data(data):
    X = []
    y = []

    for row in data:
        shape = row[0].shape
        normalized_with_bias = list((row[0] / 255.).reshape(1, shape[0] * shape[1])[0])
        normalized_with_bias = [1.] + normalized_with_bias
        X.append(np.array(normalized_with_bias))
        y.append(row[1])

    return X, y

def deprocess_data(X, y):
    data = []
    for img, label in zip(X, y):
        img_new = img[1:]
        img_shape = int(img_new.shape[0] ** 0.5)
        img_new = img_new.reshape((img_shape, img_shape))
        data.append((img_new, label[0]))
    return data

class OneLayerNN(BaseEstimator, ClassifierMixin):
    def __init__(self, classes, lr=1e-3, stop=10000):
        self.lr = lr
        self.stop = stop
        self.w = 0
        self.f = OneLayerNN.sigmoid
        self.gradf = OneLayerNN.dsigmoid
        self.classes_ = labels

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))
    
    @staticmethod
    def dsigmoid(x):
        f = OneLayerNN.sigmoid(x)
        return f * (1 - f)

    def fit(self, X, y):
        n = X.shape[1]
        self.w = np.random.randn(n, y.shape[1])
        
        # unique_labels(y)
    
        epoch = 0
        while True:
            v = np.dot(X, self.w)
            y_hat = self.f(v)
            e = (y_hat - y)
            g = np.dot(X.T, (np.multiply(e, self.gradf(v))))
            self.w = self.w - self.lr * g
            E = np.sum(e ** 2)
            if epoch > self.stop :
                break
            epoch += 1

    def predict(self, X):
        return np.int16(np.round(self.f(np.dot(X, self.w))))

data, labels = read_data()
X, y = preprocess_data(data)

X = np.array(X)
y = np.array(y).reshape((-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y)

model = OneLayerNN(classes=labels)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

data = deprocess_data(X_test, y_pred)
plot_ims(data, labels)
disp = plot_confusion_matrix(model, X_test, y_test)
plt.show()
