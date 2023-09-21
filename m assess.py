
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


samples_predictions = model.predict(x_train_mixed)  # model是训练完成后的模型；x_train_mixed是训练集
print(samples_predictions)
print(samples_predictions.shape)

samples_predictions_data = samples_predictions[:,0]
print(samples_predictions_data)
plt.hist(samples_predictions_data,bins=200,edgecolor="black",histtype="step",alpha=1,log=True,label="x_train_mixed")
plt.xlabel("Prediction Probability")
plt.ylabel("Sample Count")
plt.legend()
plt.show()
plt.hist(samples_predictions_data,bins=200,edgecolor="black",histtype="step",alpha=1,label="x_train_mixed")
plt.xlabel("Prediction Probability")
plt.ylabel("Sample Count")
plt.legend()
plt.show()

signal_threshold = 0.540    # 预测直方图的谷值

x_train_2 = sample2.reshape(sample2.shape[0], 1000, 1).astype(np.float32)
samples_predictions = model.predict(x_train_2)
print(samples_predictions)
print(samples_predictions.shape)
samples_predictions_data = samples_predictions[:,0]
print(samples_predictions_data)

Prediction_labels= np.where(signal_threshold < samples_predictions_data, 1, 0)
signal = np.bincount(Prediction_labels == 1)[1]
background = np.bincount(Prediction_labels == 0)[1]
print('Signal proportion: {:.2f}%'.format(signal / len(Prediction_labels) * 100))
print('Background proportion: {:.2f}%'.format(background / len(Prediction_labels) * 100))

cm = confusion_matrix(sample2_labels, Prediction_labels)  #  sample2_labels是填充电荷直方图划分的标签
noise_acc = cm[0, 0] / np.sum(cm[0])
signal_acc = cm[1, 1] / np.sum(cm[1])
print('Noise accuracy:', noise_acc)
print('Signal accuracy:', signal_acc)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Noise', 'Signal'], yticklabels=['Noise', 'Signal'])
plt.title('Confusion Matrix')
plt.ylabel('True labels')
plt.show()