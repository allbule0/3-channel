# 扩展版块：用于观察两个训练数据集的信号占比，验证两者之间的信号比例是否达到要求

import tensorflow as tf
import torch
import numpy as np
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.cuda_version)
print(torch.backends.cudnn.version())
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ch3 sample2 查看按照不同信号比例制作的两个训练集的signal ratio 是否达到要求
with open('ch3_sample2_o1_0.65.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample_o1 = np.array(data)
    print(sample_o1)
    print(sample_o1.shape)
with open('ch3_sample2_o2_0.35.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample_o2 = np.array(data)
    print(sample_o2)
    print(sample_o2.shape)
with open('ch3_sample2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample2 = np.array(data)
    print(sample2)
    print(sample2.shape)

import matplotlib.pyplot as plt

sample2_charge = sample2.sum(axis=1)
plt.figure()
plt.title("ch3")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample2_charge,bins=1000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_1 = 226
sample27_labels = np.where(signal_threshold_1 < sample2_charge, 1, 0)
print(sample27_labels)
print(sample27_labels.shape)
signal = np.bincount(sample27_labels == 1)[1]
background = np.bincount(sample27_labels == 0)[1]
print('ch7 Signal proportion: {:.2f}%'.format(signal / len(sample27_labels) * 100))
print('ch7 Background proportion: {:.2f}%'.format(background / len(sample27_labels) * 100))

sample_o1_charge = sample_o1.sum(axis=1)
plt.figure()
plt.title("ch3")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o1_charge,bins=1000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_2 = 210
sample_o1_labels = np.where(signal_threshold_2 < sample_o1_charge, 1, 0)
print(sample_o1_labels)
print(sample_o1_labels.shape)
signal = np.bincount(sample_o1_labels == 1)[1]
background = np.bincount(sample_o1_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample_o1_labels) * 100))
print('sample_o1 Background proportion: {:.2f}%'.format(background / len(sample_o1_labels) * 100))

sample_o2_charge = sample_o2.sum(axis=1)
plt.figure()
plt.title("ch3")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o2_charge,bins=1000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_3 = 210
sample_o2_labels = np.where(signal_threshold_3 < sample_o2_charge, 1, 0)
print(sample_o2_labels)
print(sample_o2_labels.shape)
signal = np.bincount(sample_o2_labels == 1)[1]
background = np.bincount(sample_o2_labels == 0)[1]
print('sample_o2 Signal proportion: {:.2f}%'.format(signal / len(sample_o2_labels) * 100))
print('sample_o2 Background proportion: {:.2f}%'.format(background / len(sample_o2_labels) * 100))


# ch3 sample2 查看两个训练集的比例
with open('ch3_sample_o1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample_o1 = np.array(data)
    print(sample_o1)
    print(sample_o1.shape)
with open('ch3_sample_o2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample_o2 = np.array(data)
    print(sample_o2)
    print(sample_o2.shape)
with open('ch3_sample2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample2 = np.array(data)
    print(sample2)
    print(sample2.shape)


import matplotlib.pyplot as plt

sample2_charge = sample2.sum(axis=1)
plt.figure()
plt.title("ch3")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample2_charge,bins=1000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_1 = 226
sample27_labels = np.where(signal_threshold_1 < sample2_charge, 1, 0)
print(sample27_labels)
print(sample27_labels.shape)
signal = np.bincount(sample27_labels == 1)[1]
background = np.bincount(sample27_labels == 0)[1]
print('ch7 Signal proportion: {:.2f}%'.format(signal / len(sample27_labels) * 100))
print('ch7 Background proportion: {:.2f}%'.format(background / len(sample27_labels) * 100))

sample_o1_charge = sample_o1.sum(axis=1)
plt.figure()
plt.title("ch3")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o1_charge,bins=1000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_2 = 226
sample_o1_labels = np.where(signal_threshold_2 < sample_o1_charge, 1, 0)
print(sample_o1_labels)
print(sample_o1_labels.shape)
signal = np.bincount(sample_o1_labels == 1)[1]
background = np.bincount(sample_o1_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample_o1_labels) * 100))
print('sample_o17 Background proportion: {:.2f}%'.format(background / len(sample_o1_labels) * 100))

sample_o2_charge = sample_o2.sum(axis=1)
plt.figure()
plt.title("ch3")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o2_charge,bins=1000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_3 = 221
sample_o2_labels = np.where(signal_threshold_3 < sample_o2_charge, 1, 0)
print(sample_o2_labels)
print(sample_o2_labels.shape)
signal = np.bincount(sample_o2_labels == 1)[1]
background = np.bincount(sample_o2_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample_o2_labels) * 100))
print('sample_o1 Background proportion: {:.2f}%'.format(background / len(sample_o2_labels) * 100))


# ch3 sample1 查看两个训练集的比例
with open('ch3_sample1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1 = np.array(data)
    print(sample1)
    print(sample1.shape)
with open('ch3_sample1_o1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1_o1 = np.array(data)
    print(sample1_o1)
    print(sample1_o1.shape)
with open('ch3_sample1_o2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1_o2 = np.array(data)
    print(sample1_o2)
    print(sample1_o2.shape)

sample1_charge = sample1.sum(axis=1)
plt.figure()
plt.title("ch3")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample1_charge,bins=1000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_1 = 226
sample17_labels = np.where(signal_threshold_1 < sample1_charge, 1, 0)
print(sample17_labels)
print(sample17_labels.shape)
signal = np.bincount(sample17_labels == 1)[1]
background = np.bincount(sample17_labels == 0)[1]
print('ch3 Signal proportion: {:.2f}%'.format(signal / len(sample17_labels) * 100))
print('ch3 Background proportion: {:.2f}%'.format(background / len(sample17_labels) * 100))

sample1_o1_charge = sample1_o1.sum(axis=1)
plt.figure()
plt.title("ch3")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample1_o1_charge,bins=1000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample1_o1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_2 = 242
sample1_o1_labels = np.where(signal_threshold_2 < sample1_o1_charge, 1, 0)
print(sample1_o1_labels)
print(sample1_o1_labels.shape)
signal = np.bincount(sample1_o1_labels == 1)[1]
background = np.bincount(sample1_o1_labels == 0)[1]
print('sample1_o1 Signal proportion: {:.2f}%'.format(signal / len(sample1_o1_labels) * 100))
print('sample1_o1 Background proportion: {:.2f}%'.format(background / len(sample1_o1_labels) * 100))

sample1_o2_charge = sample1_o2.sum(axis=1)
plt.figure()
plt.title("ch3")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample1_o2_charge,bins=1000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample1_o2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_3 = 355
sample1_o2_labels = np.where(signal_threshold_3 < sample1_o2_charge, 1, 0)
print(sample1_o2_labels)
print(sample1_o2_labels.shape)
signal = np.bincount(sample1_o2_labels == 1)[1]
background = np.bincount(sample1_o2_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample1_o2_labels) * 100))
print('sample_o17 Background proportion: {:.2f}%'.format(background / len(sample1_o2_labels) * 100))


# ch7 sample2 查看两个训练集的比例
with open('ch7_sample_o1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample_o1 = np.array(data)
    print(sample_o1)
    print(sample_o1.shape)
with open('ch7_sample_o2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample_o2 = np.array(data)
    print(sample_o2)
    print(sample_o2.shape)
with open('ch7_sample2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample2 = np.array(data)
    print(sample2)
    print(sample2.shape)


import matplotlib.pyplot as plt

sample2_charge = sample2.sum(axis=1)
plt.figure()
plt.title("ch7")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample2_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_1 = 270
sample27_labels = np.where(signal_threshold_1 < sample2_charge, 1, 0)
print(sample27_labels)
print(sample27_labels.shape)
signal = np.bincount(sample27_labels == 1)[1]
background = np.bincount(sample27_labels == 0)[1]
print('ch7 Signal proportion: {:.2f}%'.format(signal / len(sample27_labels) * 100))
print('ch7 Background proportion: {:.2f}%'.format(background / len(sample27_labels) * 100))

sample_o1_charge = sample_o1.sum(axis=1)
plt.figure()
plt.title("ch7")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o1_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_2 = 240
sample_o1_labels = np.where(signal_threshold_2 < sample_o1_charge, 1, 0)
print(sample_o1_labels)
print(sample_o1_labels.shape)
signal = np.bincount(sample_o1_labels == 1)[1]
background = np.bincount(sample_o1_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample_o1_labels) * 100))
print('sample_o1 Background proportion: {:.2f}%'.format(background / len(sample_o1_labels) * 100))

sample_o2_charge = sample_o2.sum(axis=1)
plt.figure()
plt.title("ch7")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o2_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_3 = 298
sample_o2_labels = np.where(signal_threshold_3 < sample_o2_charge, 1, 0)
print(sample_o2_labels)
print(sample_o2_labels.shape)
signal = np.bincount(sample_o2_labels == 1)[1]
background = np.bincount(sample_o2_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample_o2_labels) * 100))
print('sample_o1 Background proportion: {:.2f}%'.format(background / len(sample_o2_labels) * 100))


# ch7 sample1 查看两个训练集的比例
with open('ch7_sample1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1 = np.array(data)
    print(sample1)
    print(sample1.shape)
with open('ch7_sample1_o1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1_o1 = np.array(data)
    print(sample1_o1)
    print(sample1_o1.shape)
with open('ch7_sample1_o2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1_o2 = np.array(data)
    print(sample1_o2)
    print(sample1_o2.shape)

sample1_charge = sample1.sum(axis=1)
plt.figure()
plt.title("ch7")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample1_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_1 = 192
sample17_labels = np.where(signal_threshold_1 < sample1_charge, 1, 0)
print(sample17_labels)
print(sample17_labels.shape)
signal = np.bincount(sample17_labels == 1)[1]
background = np.bincount(sample17_labels == 0)[1]
print('ch3 Signal proportion: {:.2f}%'.format(signal / len(sample17_labels) * 100))
print('ch3 Background proportion: {:.2f}%'.format(background / len(sample17_labels) * 100))

sample1_o1_charge = sample1_o1.sum(axis=1)
plt.figure()
plt.title("ch7")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample1_o1_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample1_o1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_2 = 261
sample1_o1_labels = np.where(signal_threshold_2 < sample1_o1_charge, 1, 0)
print(sample1_o1_labels)
print(sample1_o1_labels.shape)
signal = np.bincount(sample1_o1_labels == 1)[1]
background = np.bincount(sample1_o1_labels == 0)[1]
print('sample1_o1 Signal proportion: {:.2f}%'.format(signal / len(sample1_o1_labels) * 100))
print('sample1_o1 Background proportion: {:.2f}%'.format(background / len(sample1_o1_labels) * 100))

sample1_o2_charge = sample1_o2.sum(axis=1)
plt.figure()
plt.title("ch7")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample1_o2_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample1_o2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_3 = 220
sample1_o2_labels = np.where(signal_threshold_3 < sample1_o2_charge, 1, 0)
print(sample1_o2_labels)
print(sample1_o2_labels.shape)
signal = np.bincount(sample1_o2_labels == 1)[1]
background = np.bincount(sample1_o2_labels == 0)[1]
print('sample1_o1 Signal proportion: {:.2f}%'.format(signal / len(sample1_o2_labels) * 100))
print('sample1_o1 Background proportion: {:.2f}%'.format(background / len(sample1_o2_labels) * 100))


# ch63 sample2 查看两个训练集的比例
with open('ch63_sample_o1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample_o1 = np.array(data)
    print(sample_o1)
    print(sample_o1.shape)
with open('ch63_sample_o2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample_o2 = np.array(data)
    print(sample_o2)
    print(sample_o2.shape)
with open('ch63_sample2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample2 = np.array(data)
    print(sample2)
    print(sample2.shape)

import matplotlib.pyplot as plt

sample2_charge = sample2.sum(axis=1)
plt.figure()
plt.title("ch63")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample2_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_1 = 290
sample27_labels = np.where(signal_threshold_1 < sample2_charge, 1, 0)
print(sample27_labels)
print(sample27_labels.shape)
signal = np.bincount(sample27_labels == 1)[1]
background = np.bincount(sample27_labels == 0)[1]
print('ch63 Signal proportion: {:.2f}%'.format(signal / len(sample27_labels) * 100))
print('ch63 Background proportion: {:.2f}%'.format(background / len(sample27_labels) * 100))

sample_o1_charge = sample_o1.sum(axis=1)
plt.figure()
plt.title("ch63")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o1_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_2 = 298
sample_o1_labels = np.where(signal_threshold_2 < sample_o1_charge, 1, 0)
print(sample_o1_labels)
print(sample_o1_labels.shape)
signal = np.bincount(sample_o1_labels == 1)[1]
background = np.bincount(sample_o1_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample_o1_labels) * 100))
print('sample_o1 Background proportion: {:.2f}%'.format(background / len(sample_o1_labels) * 100))

sample_o2_charge = sample_o2.sum(axis=1)
plt.figure()
plt.title("ch63")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o2_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o2")
plt.xlim([0, 10000])
plt.legend()
plt.show()
signal_threshold_3 = 250
sample_o2_labels = np.where(signal_threshold_3 < sample_o2_charge, 1, 0)
print(sample_o2_labels)
print(sample_o2_labels.shape)
signal = np.bincount(sample_o2_labels == 1)[1]
background = np.bincount(sample_o2_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample_o2_labels) * 100))
print('sample_o1 Background proportion: {:.2f}%'.format(background / len(sample_o2_labels) * 100))


# ch63 sample1 查看两个训练集的比例
with open('ch63_sample1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1 = np.array(data)
    print(sample1)
    print(sample1.shape)
with open('ch63_sample1_o1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1_o1 = np.array(data)
    print(sample1_o1)
    print(sample1_o1.shape)
with open('ch63_sample1_o2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1_o2 = np.array(data)
    print(sample1_o2)
    print(sample1_o2.shape)

sample1_charge = sample1.sum(axis=1)
plt.figure()
plt.title("ch63")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample1_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_1 = 245
sample17_labels = np.where(signal_threshold_1 < sample1_charge, 1, 0)
print(sample17_labels)
print(sample17_labels.shape)
signal = np.bincount(sample17_labels == 1)[1]
background = np.bincount(sample17_labels == 0)[1]
print('ch63 Signal proportion: {:.2f}%'.format(signal / len(sample17_labels) * 100))
print('ch63 Background proportion: {:.2f}%'.format(background / len(sample17_labels) * 100))

sample1_o1_charge = sample1_o1.sum(axis=1)
plt.figure()
plt.title("ch63")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample1_o1_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample1_o1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
signal_threshold_2 = 290
sample1_o1_labels = np.where(signal_threshold_2 < sample1_o1_charge, 1, 0)
print(sample1_o1_labels)
print(sample1_o1_labels.shape)
signal = np.bincount(sample1_o1_labels == 1)[1]
background = np.bincount(sample1_o1_labels == 0)[1]
print('sample1_o1 Signal proportion: {:.2f}%'.format(signal / len(sample1_o1_labels) * 100))
print('sample1_o1 Background proportion: {:.2f}%'.format(background / len(sample1_o1_labels) * 100))

sample1_o2_charge = sample1_o2.sum(axis=1)
plt.figure()
plt.title("ch63")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample1_o2_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample1_o2")
plt.xlim([0, 10000])
plt.legend()
plt.show()
signal_threshold_3 = 258
sample1_o2_labels = np.where(signal_threshold_3 < sample1_o2_charge, 1, 0)
print(sample1_o2_labels)
print(sample1_o2_labels.shape)
signal = np.bincount(sample1_o2_labels == 1)[1]
background = np.bincount(sample1_o2_labels == 0)[1]
print('sample1_o1 Signal proportion: {:.2f}%'.format(signal / len(sample1_o2_labels) * 100))
print('sample1_o1 Background proportion: {:.2f}%'.format(background / len(sample1_o2_labels) * 100))