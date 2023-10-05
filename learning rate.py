# 扩展版块：介绍三种变化的学习率方法

# 第一种
# 指定周期降低学习率
from keras.callbacks import LearningRateScheduler
# 定义学习率调度器函数
def lr_scheduler(epoch):
    # 初始学习率
    initial_lr = 0.01  # 以下参数均根据需求设定

    # 衰减学习率的速率
    decay_factor = 0.5

    # 每隔一定周期衰减学习率
    if epoch % 10 == 0 and epoch > 0:
        new_lr = initial_lr * (decay_factor ** (epoch // 10))
        return new_lr
    else:
        return initial_lr

# 创建学习率调度器
lr_callback = LearningRateScheduler(lr_scheduler)

# 创建模型
model = mcnndep(inputshape, filterAg, classnum)

# 编译模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型时添加学习率调度器
conv_hist = model.fit(
    x_train_mixed, y_train_mixed,
    batch_size=128, epochs=30,
    validation_data=(x_test_mixed, y_test_mixed),
    callbacks=[lr_callback]
)


# 第二种
# 连续几个周期损失没有改善时降低学习率
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
# 编译模型
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"]) # 设定起始学习率

# 创建ReduceLROnPlateau回调函数
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',  # 监控指标：验证集损失
    factor=0.5,  # 学习率降低的因子，可以根据需要调整
    patience=3,  # 连续几个周期损失没有改善时降低学习率
    min_lr=1e-6,  # 学习率下限
    verbose=1  # 显示学习率变化信息
)

# 训练模型时添加ReduceLROnPlateau回调函数
history = model.fit(
    x_train_mixed, y_train_mixed,
    batch_size=128,
    epochs=30,
    validation_data=(x_test_mixed, y_test_mixed),
    callbacks=[reduce_lr_callback]
)


# 第三种
# 每五个周期降低一次学习率
from tensorflow.keras.optimizers import Adam
# 自定义学习率调度器
class CustomLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, initial_lr, schedule):
        super().__init__()
        self.initial_lr = initial_lr
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        # 计算当前学习率
        lr = self.schedule(epoch, self.initial_lr)
        # 设置当前学习率到优化器
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"\nEpoch {epoch + 1}: Learning rate is {lr:.6f}")

# 定义学习率调度函数
def lr_schedule(epoch, initial_lr):
    # 根据需求定义学习率调度规则
    if epoch < 5:
        return initial_lr
    elif epoch < 10:
        return initial_lr * 0.1
    elif epoch < 15:
        return initial_lr * 0.05
    elif epoch < 20:
        return initial_lr * 0.01
    elif epoch < 25:
        return initial_lr * 0.005
    else:
        return initial_lr * 0.001


# 编译模型
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

# 初始化自定义学习率调度器
initial_lr = 0.01  # 设置初始学习率
lr_scheduler = CustomLearningRateScheduler(initial_lr, lr_schedule)

# 模型训练
conv_hist = model.fit(
    x_train_mixed, y_train_mixed,
    batch_size=128, epochs=30,  # 根据你的需求设置训练周期
    validation_data=(x_test_mixed, y_test_mixed),
    callbacks=[lr_scheduler],  # 添加自定义学习率调度器回调
)