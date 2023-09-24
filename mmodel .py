# 接在m data processing.py文件后
# 模型版块：一维深度卷积模型

from keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint


inputshape =(1000,1)
filterAg = {'cnn_l1': 32, 'cnn_l2': 64, 'cnn_l3': 64, 'cnn_l4': 128}
classnum = 2
def mcnndep(inputshape, filterAg, classnum):
    x = Input(inputshape)
    inputs = x

    x = Conv1D(filters=filterAg['cnn_l1'], kernel_size=15, strides=2,
        activation='relu', kernel_initializer=glorot_uniform(seed=0), name="cnv1")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
    print("Stage :1", x)

    x = Conv1D(filters=filterAg['cnn_l2'], kernel_size=3, strides=2, 
        activation='relu', kernel_initializer=glorot_uniform(seed=0), name="cnv2")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
    print("Stage :2", x)

    x = Conv1D(filters=filterAg['cnn_l3'], kernel_size=3, strides=2, 
        activation='relu', kernel_initializer=glorot_uniform(seed=0), name="cnv3")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
    print("Stage :3", x)

    x = Conv1D(filters=filterAg['cnn_l4'], kernel_size=3, strides=2, 
        activation='relu', kernel_initializer=glorot_uniform(seed=0), name="cnv4")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
    print("Stage :4", x)

    x = Flatten()(x)
    print("Stage :5", x)

    x = Dense(units=classnum, activation="softmax", kernel_initializer=glorot_uniform(seed=0), name="den1")(x)
    print("Stage :6", x)

    return Model(inputs=inputs, outputs=x, name="mcnndep")

model = mcnndep(inputshape, filterAg, classnum)
model.summary()


# 添加回调，保存最佳模型权重，用于模型预测，如果不用可以去掉
checkpoint = ModelCheckpoint("ch3 best_model_weights-1.h5", save_best_only=True,
                             monitor="val_loss", mode="min", verbose=1)


# 编译模型 ，指定学习率
from tensorflow.keras.optimizers import Adam

custom_learning_rate = 0.02
optimizer = Adam(learning_rate=custom_learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# 如果不使用指定学习率，默认使用adam,可将其余删除，只保留编译语句，且optimizer=“adam"
current_learning_rate = optimizer.learning_rate.numpy()
print("当前学习率:", current_learning_rate)


# 训练模型
conv_hist = model.fit(x_train_mixed, y_train_mixed, batch_size=128, epochs=30,
                      validation_data=(x_test_mixed,y_test_mixed), callbacks=[checkpoint])
# x_train_mixed, y_train_mixed是训练集和训练标签；x_test_mixed,y_test_mixed是验证集和验证标签
# 如果不使用回调，去除最后的callbacks=[checkpoint]

# 训练过程可视化
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title("Convolution Loss")
plt.plot(conv_hist.history["loss"], label="loss")
plt.plot(conv_hist.history["val_loss"], label="val_loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Convolution Accuracy")
plt.plot(conv_hist.history["accuracy"], label="accuracy")
plt.plot(conv_hist.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.show()