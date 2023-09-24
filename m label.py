# 扩展版块：填充电荷直方图制备类别标签

import numpy as np
import matplotlib.pyplot as plt

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

sample2_charge = sample2.sum(axis=1)

plt.figure()
plt.title("ch7")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample2_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample2")
plt.xlim([0, 1000])
plt.legend()
plt.show()

sample2_signal_threshold = 270                   # 填充电荷直方图的谷值
sample2_labels = np.where(sample2_signal_threshold < sample2_charge, 1, 0)
print(sample2_labels)
print(sample2_labels.shape)

signal = np.bincount(sample2_labels == 1)[1]
background = np.bincount(sample2_labels == 0)[1]
print('ch7 Signal proportion: {:.2f}%'.format(signal / len(sample2_labels) * 100))
print('ch7 Background proportion: {:.2f}%'.format(background / len(sample2_labels) * 100))

# 以下重复
sample_o1_charge = sample_o1.sum(axis=1)
plt.figure()
plt.title("ch7")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o1_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o1")
plt.xlim([0, 1000])
plt.legend()
plt.show()
sample_o1_signal_threshold = 270
sample_o1_labels = np.where(sample_o1_signal_threshold < sample_o1_charge, 1, 0)
print(sample_o1_labels)
print(sample_o1_labels.shape)
signal = np.bincount(sample_o1_labels == 1)[1]
background = np.bincount(sample_o1_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample_o1_labels) * 100))
print('sample_o17 Background proportion: {:.2f}%'.format(background / len(sample_o1_labels) * 100))

sample_o2_charge = sample_o2.sum(axis=1)
plt.figure()
plt.title("ch7")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_o2_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o2")
plt.xlim([0, 1000])
plt.legend()
plt.show()
sample_o2_signal_threshold = 290
sample_o2_labels = np.where(sample_o2_signal_threshold < sample_o2_charge, 1, 0)
print(sample_o2_labels)
print(sample_o2_labels.shape)
signal = np.bincount(sample_o2_labels == 1)[1]
background = np.bincount(sample_o2_labels == 0)[1]
print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample_o2_labels) * 100))
print('sample_o17 Background proportion: {:.2f}%'.format(background / len(sample_o2_labels) * 100))