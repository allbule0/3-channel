# 扩展版块：画出彩色两类直方图，方便判断模型性能的好坏，以及直观查看预测结果不一致的波形


import numpy as np
import matplotlib.pyplot as plt

# 导入数据。可根据需要更换成其他样本集
x_train_mixed = np.concatenate([sample_o1_1, sample_o2_2], axis=0)  # sample_o1_1是第一个训练样本集；sample_o2_2是另一个训练样本集
train = x_train_mixed
sample_charge = train.sum(axis=1)
charge_threshold = min_value   # min_value是电荷谱的阈值，通常为谷底
samples_predictions = model.predict(x_train_mixed)  # model是训练完成后的模型
samples_predictions_data = samples_predictions[:,0]

# 第一种图： model vs charge
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.title("model")
plt.hist(samples_predictions_data[:len(sample_o1_1)], bins=100, edgecolor="black", histtype="bar", alpha=0.5, log=True, label="sample A", color="blue")
plt.hist(samples_predictions_data[len(sample_o1_1):], bins=100, edgecolor="black", histtype="bar", alpha=0.2, log=True, label="sample B", color="red")
plt.xlabel("Prediction Probability")
plt.ylabel("Sample Count")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("charge")
plt.xlabel('Electric charge')
plt.ylabel("Sample Count")
plt.hist(sample_charge[sample_charge < charge_threshold], bins=100, edgecolor="black", histtype="bar", log=True, alpha=0.3, label="background", color="red")
plt.hist(sample_charge[sample_charge >= charge_threshold], bins=4000, edgecolor="black", histtype="bar", log=True, alpha=0.5, label="signal", color="blue")
plt.xlim([0,1000])
plt.legend()
plt.show()

# 第二种图：training sample的log形式和无log形式
plt.hist(samples_predictions_data[:len(sample_o1_1)], bins=100, edgecolor="black", histtype="bar", alpha=0.5, log=True, label="sample A", color="blue")
plt.hist(samples_predictions_data[len(sample_o1_1):], bins=100, edgecolor="black", histtype="bar", alpha=0.3, log=True, label="sample B", color="red")
plt.xlabel("Prediction Probability")
plt.ylabel("Sample Count")
plt.legend()
plt.show()
plt.hist(samples_predictions_data[:len(sample_o1_1)], bins=100, edgecolor="black", histtype="bar", alpha=0.5, label="sample A", color="blue")
plt.hist(samples_predictions_data[len(sample_o1_1):], bins=100, edgecolor="black", histtype="bar", alpha=0.3, label="sample B", color="red")
plt.xlabel("Prediction Probability")
plt.ylabel("Sample Count")
plt.legend()
plt.show()

# 导入数据，与前面一样，只是将training sample换成了另一个样本集sample2
x_train_2 = sample2.reshape(sample2.shape[0],1000, 1).astype(np.float32)
sample2_predictions = model.predict(x_train_2)
sample2_predictions_data = sample2_predictions[:,0]
sample2_charge = sample2.sum(axis=1)
sample2_signal_threshold = min_value             # 填充电荷直方图的谷值

# 第三种图：sample2预测结果根据电荷区分类别画成两类的直方图
plt.title("model")
plt.xlabel("Prediction Probability")
plt.ylabel("Sample Count")
plt.hist(sample2_predictions_data[sample2_charge < sample2_signal_threshold], bins=100, edgecolor="black", histtype="bar", log=True, alpha=0.3, label="background", color="red")
plt.hist(sample2_predictions_data[sample2_charge >= sample2_signal_threshold], bins=100, edgecolor="black", histtype="bar", log=True, alpha=0.5, label="signal", color="blue")
plt.legend()
plt.show()
