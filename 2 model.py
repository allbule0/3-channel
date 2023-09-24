from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.models import Model
import matplotlib.pyplot as plt

inputshape =(40,25,1)
filterAg = {'cnn_l1': 32, 'cnn_l2': 64, 'cnn_l3': 64, 'cnn_l4': 128}
classnum = 2


def mcnndep(inputshape, filterAg, classnum):
    x = Input(inputshape)
    inputs = x

    x = Conv2D(filters=filterAg['cnn_l1'], kernel_size=(3, 3), strides=2,
        activation='relu', kernel_initializer=glorot_uniform(seed=0), name="cnv1")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(x)
    print("Stage :1", x)

    x = Conv2D(filters=filterAg['cnn_l2'], kernel_size=(3, 3), strides=2,
        activation='relu', kernel_initializer=glorot_uniform(seed=0), name="cnv2")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(x)
    print("Stage :2", x)

    x = Conv2D(filters=filterAg['cnn_l3'], kernel_size=(3, 3), strides=2,
        activation='relu', kernel_initializer=glorot_uniform(seed=0), name="cnv3")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(x)
    print("Stage :3", x)

    x = Flatten()(x)
    print("Stage :5", x)

    x = Dense(units=classnum, activation="softmax", kernel_initializer=glorot_uniform(seed=0), name="den1")(x)
    print("Stage :6", x)

    return Model(inputs=inputs, outputs=x, name="mcnndep")


model = mcnndep(inputshape, filterAg, classnum)
model.summary()
quit(0)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
conv_hist = model.fit(x_train_mixed, y_train_mixed, batch_size=128, epochs=50, validation_data=(x_test_mixed,y_test_mixed))
# x_train_mixed, y_train_mixed是训练集和训练标签；x_test_mixed,y_test_mixed是验证集和验证标签

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