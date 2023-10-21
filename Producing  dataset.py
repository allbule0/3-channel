# 扩展版块：介绍四种制备不同信号比例的训练数据集的方法


import numpy as np
import matplotlib.pyplot as plt
import random

print("第一种")
# 第一种
# 以填充波形电荷的方法，初步赋予每个波形标签（signal or background)，
# 根据要求的signal ratio ，选取指定数量的信号波形，并用sample0（纯噪声样本）补充数量
# with open('ch3_sample2.txt', 'r') as f:
#     data = f.readlines()
#     data = [list(map(float, line.strip().split())) for line in data]
#     sample2 = np.array(data)
#     print(sample2)
#     print(sample2.shape)
# with open('ch3_sample0.txt', 'r') as f:
#     data = f.readlines()
#     data = [list(map(float, line.strip().split())) for line in data]
#     sample0 = np.array(data)
#     print(sample0)
#     print(sample0.shape)
#
# # 填充电荷，划阈值赋予标签
# sample2_charge = sample2.sum(axis=1)
# plt.figure()
# plt.title("ch63")
# plt.xlabel('Electric charge')
# plt.ylabel("Sample Count")
# plt.hist(sample2_charge,bins=10000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample2")
# plt.xlim([150, 400])
# plt.legend()
# plt.show()
#
# signal_threshold_1 = min_value  # 阈值选择最低点
# sample2_labels = np.where(signal_threshold_1 < sample2_charge, 1, 0)
# print(sample2_labels)
# print(sample2_labels.shape)
# signal = np.bincount(sample2_labels == 1)[1]
# background = np.bincount(sample2_labels == 0)[1]
# print('ch3 Signal proportion: {:.2f}%'.format(signal / len(sample2_labels) * 100))
# print('ch3 Background proportion: {:.2f}%'.format(background / len(sample2_labels) * 100))
#
# # 提取出标签为1的波形样本
# signal_waveform = sample2[sample2_labels == 1]
# print(signal_waveform)
# print(signal_waveform.shape)
#
# background_waveform = sample2[sample2_labels == 0]
# print(background_waveform)
# print(background_waveform.shape)
#
# # 制备训练数据集
# random_indices_o1_1 = random.sample(range(len(signal_waveform)), 12345)   # 16250是我想选取的信号波形，可根据所需更改
# sample2_o1_1 = signal_waveform[random_indices_o1_1]
# random_indices_o1_2 = random.sample(range(len(sample0)), 4000)        # 8750是我想补充的背景波形，可根据所需更改
# sample2_o1_2 = sample0[random_indices_o1_2]
# sample2_o1 = np.concatenate((sample2_o1_1, sample2_o1_2), axis=0)
# print(sample2_o1)         # 得到一个训练数据集
# print(sample2_o1.shape)
#
# remaining_indices_signal = list(set(range(len(signal_waveform))) - set(random_indices_o1_1))
# remaining_indices_sample0 = list(set(range(len(sample0))) - set(random_indices_o1_2))
# random_indices_o2_1 = random.sample(remaining_indices_signal, 3210)   # 8750是我想选取的信号波形，可根据所需更改
# sample2_o2_1 = signal_waveform[random_indices_o2_1]
# random_indices_o2_2 = random.sample(remaining_indices_sample0, 13000)   # 16250是我想补充的背景波形，可根据所需更改
# sample2_o2_2 = sample0[random_indices_o2_2]
# sample2_o2 = np.concatenate((sample2_o2_1, sample2_o2_2), axis=0)
# print(sample2_o2)          # 得到另一个训练数据集
# print(sample2_o2.shape)
#
# # 画数据集的平均波形图
# sample01_waveform = np.mean(signal_waveform, axis=0)
# sample02_waveform = np.mean(background_waveform, axis=0)
# plt.figure(figsize=(16, 5))
# plt.subplot(1, 2, 1)
# plt.plot(sample01_waveform,label='signal waveform')
# plt.title('Average Waveform')
# plt.xlabel('time[ns]')
# plt.ylabel('amplitude')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(sample02_waveform,label='background waveform')
# plt.title('Average Waveform')
# plt.xlabel('time[ns]')
# plt.ylabel('amplitude')
# plt.legend()
# plt.show()
# quit(0)
# # 保存数据集
# np.savetxt('ch7_sample2_1.00.txt', signal_waveform)
# np.savetxt('ch7_sample2_0.00.txt', background_waveform)
# quit(0)

print("第二种")
# # 第二种
# # 使用幅度阈值排列，再进行样本集切割成不同信号比例的两个训练数据集
# # 无需用到sample0（纯噪声样本），但是两个训练集的信号比例不容易控制
# with open('ch63_sample1.txt', 'r') as f:
#     data = f.readlines()
#     data = [list(map(float, line.strip().split())) for line in data]
#     sample1 = np.array(data)
#     print(sample1)
#     print(sample1.shape)
#
# # 按最大幅度值给波形排序并切割
# max_amplitudes = np.max(np.abs(sample1), axis=1)
# sorted_indices = np.argsort(max_amplitudes)
# length = len(sample1)
# middle_index = int(length/2)
# sample_11_indices = sorted_indices[:middle_index]
# sample_12_indices = sorted_indices[middle_index:]
# sample_11 = sample1[sample_11_indices]
# sample_12 = sample1[sample_12_indices]
# print(sample_11.shape)
# print(sample_12.shape)
#
# # 按照1/4的sample_11和3/4sample_12合成为sample_o1
# sample_11_length = len(sample_11)
# num_sample_o1 = int(sample_11_length // 4)
# print(num_sample_o1)
# random_indices_o1_11 = random.sample(range(sample_11_length), num_sample_o1)
# random_indices_o1_12 = random.sample(range(len(sample_12)), sample_11_length-num_sample_o1 )
# sample_o1_11 = sample_11[random_indices_o1_11]
# sample_o1_12 = sample_12[random_indices_o1_12]
# sample1_o1 = np.concatenate((sample_o1_11, sample_o1_12), axis=0)
# print(sample1_o1)
# print(sample1_o1.shape)
#
# # 剩余的波形作为sample_o2
# sample_11_remaining_indices = [i for i in range(sample_11_length) if i not in random_indices_o1_11]
# sample_12_remaining_indices = [i for i in range(len(sample_12)) if i not in random_indices_o1_12]
# sample_o2_11 = sample_11[sample_11_remaining_indices]
# sample_o2_12 = sample_12[sample_12_remaining_indices]
# sample1_o2 = np.concatenate((sample_o2_11, sample_o2_12), axis=0)
# print(sample1_o2)
# print(sample1_o2.shape)
#
# # 画训练数据集的平均波形图
# sample01_waveform = np.mean(sample1_o1, axis=0)
# sample02_waveform = np.mean(sample1_o2, axis=0)
# plt.figure(figsize=(16, 5))
# plt.subplot(1, 2, 1)
# plt.plot(sample01_waveform,label='sample1_o1')
# plt.title('Average Waveform')
# plt.xlabel('time[ns]')
# plt.ylabel('amplitude')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(sample02_waveform,label='sample1_o2')
# plt.title('Average Waveform')
# plt.xlabel('time[ns]')
# plt.ylabel('amplitude')
# plt.legend()
# plt.show()
#
# # 保存数据集
# np.savetxt('ch63_sample1_o1.txt', sample1_o1)
# np.savetxt('ch63_sample1_o2.txt', sample1_o2)
# quit(0)

print("第三种")
# 第三种
# 按比例随机分样本，通过sample0补充已分割的两个样本，使其接近想要的signal ratio
# with open('ch3_sample2.txt', 'r') as f:
#     data = f.readlines()
#     data = [list(map(float, line.strip().split())) for line in data]
#     sample1 = np.array(data)
#     print(sample1)
#     print(sample1.shape)
# with open('ch3_sample0.txt', 'r') as f:
#     data = f.readlines()
#     data = [list(map(float, line.strip().split())) for line in data]
#     sample0 = np.array(data)
#     print(sample0)
#     print(sample0.shape)
#
# sample3 = []
# sample4 = []
# for i in range(len(sample1)):
#     if random.random() < 0.999:
#         random_index = random.sample(range(sample1.shape[0]), 1)[0]
#         sample3.append(sample1[random_index])
#     else:
#         random_index = random.sample(range(sample1.shape[0]), 1)[0]
#         sample4.append(sample1[random_index])
#
# sample3 = np.array(sample3)
# sample4 = np.array(sample4)
# print(sample3)
# print("sample3:", sample3.shape)
# print(sample4)
# print("sample4:", sample4.shape)
#
# # # 补充部分波形
# # num1 = 30000-len(sample3)
# # random_indices_1 = random.sample(range(len(sample0)), num1)
# # sample3_0 = sample0[random_indices_1]
# # print(sample3_0.shape)
# # sample1_new1 = np.concatenate((sample3, sample3_0), axis=0)
# # print(sample1_new1)
# # print(sample1_new1.shape)
# #
# # remaining_indices_sample0 = list(set(range(len(sample0))) - set(random_indices_1))
# # num2 = 30000-len(sample4)
# # random_indices_2 = random.sample(range(len(sample0)), num2)
# # sample4_0 = sample0[random_indices_2]
# # sample1_new2 = np.concatenate((sample4, sample4_0), axis=0)
# # print(sample1_new2)
# # print(sample1_new2.shape)
#
# # 删除部分波形
# num_to_remove = len(sample3)-30000
# indices_to_remove = random.sample(range(len(sample3)), num_to_remove)
# sample1_new1 = np.delete(sample3, indices_to_remove, axis=0)
#
# # num2 = len(sample3)-len(sample4)
# # random_indices_2 = random.sample(range(len(sample0)), num2)
# # sample4_0 = sample0[random_indices_2]
# sample1_new2 = np.concatenate((sample4, sample0), axis=0)
# print(sample1_new2)
# print(sample1_new2.shape)
#
# import matplotlib.pyplot as plt
#
# sample1_charge = sample1.sum(axis=1)
# plt.figure()
# plt.title("ch3")
# plt.xlabel('Electric charge')
# plt.ylabel("Sample Count")
# plt.hist(sample1_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample2")
# plt.xlim([0, 1000])
# plt.legend()
# plt.show()
# signal_threshold_1 = 226
# sample27_labels = np.where(signal_threshold_1 < sample1_charge, 1, 0)
# print(sample27_labels)
# print(sample27_labels.shape)
# signal = np.bincount(sample27_labels == 1)[1]
# background = np.bincount(sample27_labels == 0)[1]
# print('ch7 Signal proportion: {:.2f}%'.format(signal / len(sample27_labels) * 100))
# print('ch7 Background proportion: {:.2f}%'.format(background / len(sample27_labels) * 100))
#
# sample3_charge = sample1_new1.sum(axis=1)
# plt.figure()
# plt.title("ch3")
# plt.xlabel('Electric charge')
# plt.ylabel("Sample Count")
# plt.hist(sample3_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o1")
# plt.xlim([0, 1000])
# plt.legend()
# plt.show()
# signal_threshold_2 = 226
# sample3_labels = np.where(signal_threshold_2 < sample3_charge, 1, 0)
# print(sample3_labels)
# print(sample3_labels.shape)
# signal = np.bincount(sample3_labels == 1)[1]
# background = np.bincount(sample3_labels == 0)[1]
# print('sample_o1 Signal proportion: {:.2f}%'.format(signal / len(sample3_labels) * 100))
# print('sample_o1 Background proportion: {:.2f}%'.format(background / len(sample3_labels) * 100))
#
# sample4_charge = sample1_new2.sum(axis=1)
# plt.figure()
# plt.title("ch3")
# plt.xlabel('Electric charge')
# plt.ylabel("Sample Count")
# plt.hist(sample4_charge,bins=5000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample_o2")
# plt.xlim([0, 1000])
# plt.legend()
# plt.show()
# signal_threshold_3 = 226
# sample4_labels = np.where(signal_threshold_3 < sample4_charge, 1, 0)
# print(sample4_labels)
# print(sample4_labels.shape)
# signal = np.bincount(sample4_labels == 1)[1]
# background = np.bincount(sample4_labels == 0)[1]
# print('sample_o2 Signal proportion: {:.2f}%'.format(signal / len(sample4_labels) * 100))
# print('sample_o2 Background proportion: {:.2f}%'.format(background / len(sample4_labels) * 100))
#
# # 画训练数据集的平均波形图
# sample01_waveform = np.mean(sample1_new1, axis=0)
# sample02_waveform = np.mean(sample1_new2, axis=0)
# plt.figure(figsize=(16, 5))
# plt.subplot(1, 2, 1)
# plt.plot(sample01_waveform,label='sample1_new1')
# plt.title('Average Waveform')
# plt.xlabel('time[ns]')
# plt.ylabel('amplitude')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(sample02_waveform,label='sample1_new2')
# plt.title('Average Waveform')
# plt.xlabel('time[ns]')
# plt.ylabel('amplitude')
# plt.legend()
# plt.show()
#
# # 保存数据集
# np.savetxt('ch3_sample2_new_0.999.txt', sample1_new1)
# np.savetxt('ch3_sample2_new_0.001.txt', sample1_new2)
# quit(0)

print("第四种")
# 第四种
# 粗略电荷谱阈值划分，阈值以右制作“信号样本”，以左制作“噪声样本”
# with open('ch3_sample2.txt', 'r') as f:
#     data = f.readlines()
#     data = [list(map(float, line.strip().split())) for line in data]
#     sample2 = np.array(data)
#     print(sample2)
#     print(sample2.shape)
#
# # 填充电荷，划阈值赋予标签
# sample2_charge = sample2.sum(axis=1)
# plt.figure()
# plt.title("ch3")
# plt.xlabel('Electric charge')
# plt.ylabel("Sample Count")
# plt.hist(sample2_charge,bins=10000,edgecolor="black",histtype="step",log=True,alpha=1,label="sample2")
# plt.xlim([0, 400])
# plt.legend()
# plt.show()
#
# signal_threshold_1 = min_value
# sample2_labels = np.where(signal_threshold_1 < sample2_charge, 1, 0)
# print(sample2_labels)
# print(sample2_labels.shape)
# signal = np.bincount(sample2_labels == 1)[1]
# background = np.bincount(sample2_labels == 0)[1]
# print('ch3 Signal proportion: {:.2f}%'.format(signal / len(sample2_labels) * 100))
# print('ch3 Background proportion: {:.2f}%'.format(background / len(sample2_labels) * 100))
#
# # 提取出标签为1的波形样本
# signal_waveform = sample2[sample2_labels == 1]
# print(signal_waveform)
# print(signal_waveform.shape)
#
# background_waveform = sample2[sample2_labels == 0]
# print(background_waveform)
# print(background_waveform.shape)
#
# # 画数据集的平均波形图
# sample01_waveform = np.mean(signal_waveform, axis=0)
# sample02_waveform = np.mean(background_waveform, axis=0)
# plt.figure(figsize=(16, 5))
# plt.subplot(1, 2, 1)
# plt.plot(sample01_waveform,label='signal waveform')
# plt.title('Average Waveform')
# plt.xlabel('time[ns]')
# plt.ylabel('amplitude')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(sample02_waveform,label='background waveform')
# plt.title('Average Waveform')
# plt.xlabel('time[ns]')
# plt.ylabel('amplitude')
# plt.legend()
# plt.show()
# # quit(0)
# # 保存数据集
# np.savetxt('ch3_sample2_1.00.txt', signal_waveform)
# np.savetxt('ch3_sample2_0.00.txt', background_waveform)
# quit(0)
