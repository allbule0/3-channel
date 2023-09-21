
from keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.models import Model


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


