# 扩展版块：介绍两种制备不同信号比例的训练数据集的方法


import numpy as np
import matplotlib.pyplot as plt
import random

# 第一种
# 以填充波形电荷的方法，初步赋予每个波形标签（signal or background)，
# 根据要求的signal ratio ，选取指定数量的信号波形，并用sample0（纯噪声样本）补充数量

with open('ch3_sample2.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample2 = np.array(data)
    print(sample2)
    print(sample2.shape)
with open('ch3_sample0.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample0 = np.array(data)
    print(sample0)
    print(sample0.shape)

# 填充电荷，划阈值赋予标签
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
sample2_labels = np.where(signal_threshold_1 < sample2_charge, 1, 0)
print(sample2_labels)
print(sample2_labels.shape)
signal = np.bincount(sample2_labels == 1)[1]
background = np.bincount(sample2_labels == 0)[1]
print('ch3 Signal proportion: {:.2f}%'.format(signal / len(sample2_labels) * 100))
print('ch3 Background proportion: {:.2f}%'.format(background / len(sample2_labels) * 100))

# 提取出标签为1的波形样本
signal_waveform = sample2[sample2_labels == 1]
print(signal_waveform)
print(signal_waveform.shape)

# 制备训练数据集
random_indices_o1_1 = random.sample(range(len(signal_waveform)), 16250)   # 16250是我想选取的信号波形，可根据所需更改
sample2_o1_1 = signal_waveform[random_indices_o1_1]
random_indices_o1_2 = random.sample(range(len(sample0)), 8750)        # 8750是我想补充的背景波形，可根据所需更改
sample2_o1_2 = sample0[random_indices_o1_2]
sample2_o1 = np.concatenate((sample2_o1_1, sample2_o1_2), axis=0)
print(sample2_o1)         # 得到一个训练数据集
print(sample2_o1.shape)

remaining_indices_signal = list(set(range(len(signal_waveform))) - set(random_indices_o1_1))
remaining_indices_sample0 = list(set(range(len(sample0))) - set(random_indices_o1_2))
random_indices_o2_1 = random.sample(remaining_indices_signal, 8750)   # 8750是我想选取的信号波形，可根据所需更改
sample2_o2_1 = signal_waveform[random_indices_o2_1]
random_indices_o2_2 = random.sample(remaining_indices_sample0, 16250)   # 16250是我想补充的背景波形，可根据所需更改
sample2_o2_2 = sample0[random_indices_o2_2]
sample2_o2 = np.concatenate((sample2_o2_1, sample2_o2_2), axis=0)
print(sample2_o2)          # 得到另一个训练数据集
print(sample2_o2.shape)

# 画数据集的平均波形图
sample01_waveform = np.mean(sample2_o1, axis=0)
sample02_waveform = np.mean(sample2_o2, axis=0)
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(sample01_waveform,label='sample2_o1')
plt.title('Average Waveform')
plt.xlabel('time[ns]')
plt.ylabel('amplitude')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(sample02_waveform,label='sample2_o2')
plt.title('Average Waveform')
plt.xlabel('time[ns]')
plt.ylabel('amplitude')
plt.legend()
plt.show()

# 保存数据集
np.savetxt('ch3_sample2_o1_0.65.txt', sample2_o1)
np.savetxt('ch3_sample2_o2_0.35.txt', sample2_o2)


# 第二种
# 使用幅度阈值排列，再进行样本集切割成不同信号比例的两个训练数据集
# 无需用到sample0（纯噪声样本），但是两个训练集的信号比例不容易控制
with open('ch63_sample1.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample1 = np.array(data)
    print(sample1)
    print(sample1.shape)

# 按最大幅度值给波形排序并切割
max_amplitudes = np.max(np.abs(sample1), axis=1)
sorted_indices = np.argsort(max_amplitudes)
length = len(sample1)
middle_index = int(length/2)
sample_11_indices = sorted_indices[:middle_index]
sample_12_indices = sorted_indices[middle_index:]
sample_11 = sample1[sample_11_indices]
sample_12 = sample1[sample_12_indices]
print(sample_11.shape)
print(sample_12.shape)

# 按照1/4的sample_11和3/4sample_12合成为sample_o1
sample_11_length = len(sample_11)
num_sample_o1 = int(sample_11_length // 4)
print(num_sample_o1)
random_indices_o1_11 = random.sample(range(sample_11_length), num_sample_o1)
random_indices_o1_12 = random.sample(range(len(sample_12)), sample_11_length-num_sample_o1 )
sample_o1_11 = sample_11[random_indices_o1_11]
sample_o1_12 = sample_12[random_indices_o1_12]
sample1_o1 = np.concatenate((sample_o1_11, sample_o1_12), axis=0)
print(sample1_o1)
print(sample1_o1.shape)

# 剩余的波形作为sample_o2
sample_11_remaining_indices = [i for i in range(sample_11_length) if i not in random_indices_o1_11]
sample_12_remaining_indices = [i for i in range(len(sample_12)) if i not in random_indices_o1_12]
sample_o2_11 = sample_11[sample_11_remaining_indices]
sample_o2_12 = sample_12[sample_12_remaining_indices]
sample1_o2 = np.concatenate((sample_o2_11, sample_o2_12), axis=0)
print(sample1_o2)
print(sample1_o2.shape)

# 画训练数据集的平均波形图
sample01_waveform = np.mean(sample1_o1, axis=0)
sample02_waveform = np.mean(sample1_o2, axis=0)
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(sample01_waveform,label='sample1_o1')
plt.title('Average Waveform')
plt.xlabel('time[ns]')
plt.ylabel('amplitude')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(sample02_waveform,label='sample1_o2')
plt.title('Average Waveform')
plt.xlabel('time[ns]')
plt.ylabel('amplitude')
plt.legend()
plt.show()

# 保存数据集
np.savetxt('ch63_sample1_o1.txt', sample1_o1)
np.savetxt('ch63_sample1_o2.txt', sample1_o2)