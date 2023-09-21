from tensorflow import keras
import numpy as np

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

split_idx = int(sample_o1.shape[0] * 0.2)
sample3 = sample_o1[:split_idx]
sample1 = sample_o1[split_idx:]
split_idx = int(sample_o2.shape[0] * 0.2)
sample4 = sample_o2[:split_idx]
sample2 = sample_o2[split_idx:]
x_train_mixed = np.concatenate([sample1, sample2], axis=0)
y_train_mixed = np.concatenate([np.zeros(sample1.shape[0]), np.ones(sample2.shape[0])], axis=0)
print(x_train_mixed)
print(x_train_mixed.shape)
print(y_train_mixed)
print(y_train_mixed.shape)
x_test_mixed = np.concatenate([sample3, sample4], axis=0)
y_test_mixed = np.concatenate([np.zeros(sample3.shape[0]), np.ones(sample4.shape[0])], axis=0)
print(x_test_mixed)
print(x_test_mixed.shape)
print(y_test_mixed)
print(y_test_mixed.shape)

num_classes = 2
input_shape = (1000,1)
y_train_mixed = keras.utils.to_categorical(y_train_mixed, num_classes)
print(y_train_mixed)
y_test_mixed = keras.utils.to_categorical(y_test_mixed, num_classes)
print(y_test_mixed)
x_train_mixed = x_train_mixed.reshape(x_train_mixed.shape[0], 1000, 1).astype(np.float32)
x_test_mixed = x_test_mixed.reshape(x_test_mixed.shape[0], 1000, 1).astype(np.float32)
print(x_train_mixed.shape[0], "train samples")
print(x_test_mixed.shape[0], "test samples")