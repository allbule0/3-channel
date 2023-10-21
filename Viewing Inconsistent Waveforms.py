# 扩展版块：查看不一致波形，根据人眼的判断来验证神经网络是否优于电荷谱阈值区分


import numpy as np
import matplotlib.pyplot as plt
import os

# 创建一个布尔掩码，标识哪些波形分类结果不一致
inconsistent_mask = sample_labels != Prediction_labels  # sample_labels是电荷谱阈值区分的类别；prediction-labels是模型预测区分的类别
inconsistent_indices = np.where(inconsistent_mask)[0]
print(inconsistent_indices)

inconsistent_waveform = train[inconsistent_indices]   # train是训练集，如果想查看其他样本的预测结果不一致的波形，可更改成想要的样本集
print(inconsistent_waveform.shape)
np.savetxt('ch3_train_inconsistent_waveform_cd0.txt', inconsistent_waveform)

# 画出不一致波形并自动保存文件夹
os.makedirs("ch3_train_inconsistent_waveform_images_cd0", exist_ok=True)

with open('ch3_train_inconsistent_waveform_cd0.txt', 'r') as f:
    data = f.readlines()
    data = [list(map(float, line.strip().split())) for line in data]
    sample = np.array(data)
    print(sample)
    print(sample.shape)

for i, waveform_data in enumerate(sample):
    plt.figure()
    plt.plot(waveform_data)
    plt.title(f"Inconsistent Waveform {i + 1}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    filename = os.path.join("ch3_train_inconsistent_waveform_images_cd0", f'inconsistent_waveform_{i}.png')
    plt.savefig(filename)
    plt.close()
    print(f'Saved inconsistent waveform {i} image to {filename}')