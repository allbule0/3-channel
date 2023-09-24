# 扩展版块：用于查看训练集样本的波形图

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

sample_o1_waveform = np.mean(sample_o1, axis=0)
sample_o2_waveform = np.mean(sample_o2, axis=0)
sample2_waveform = np.mean(sample2, axis=0)
plt.figure(figsize=(16, 5))
plt.subplot(1, 3, 1)
plt.plot(sample_o1_waveform,label='sample_o1')
plt.title('Average Waveform')
plt.xlabel('time[ns]')
plt.ylabel('amplitude')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(sample_o2_waveform,label='sample_o2')
plt.title('Average Waveform')
plt.xlabel('time[ns]')
plt.ylabel('amplitude')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(sample2_waveform,label='sample2')
plt.title('Average Waveform')
plt.xlabel('time[ns]')
plt.ylabel('amplitude')
plt.legend()
plt.show()