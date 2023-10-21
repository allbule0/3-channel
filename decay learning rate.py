# 扩展版块：介绍七种变化的学习率方法
# 第七种较适合本次无监督学习任务的学习


print("第一种")
# # 第一种
# # 指定周期降低学习率
# from keras.callbacks import LearningRateScheduler
# # 定义学习率调度器函数
# def lr_scheduler(epoch):
#     # 初始学习率
#     initial_lr = 0.01  # 以下参数均根据需求设定
#
#     # 衰减学习率的速率
#     decay_factor = 0.5
#
#     # 每隔一定周期衰减学习率
#     if epoch % 10 == 0 and epoch > 0:
#         new_lr = initial_lr * (decay_factor ** (epoch // 10))
#         return new_lr
#     else:
#         return initial_lr
#
# # 创建学习率调度器
# lr_callback = LearningRateScheduler(lr_scheduler)
#
# # 创建模型
# model = mcnndep(inputshape, filterAg, classnum)
#
# # 编译模型
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#
# # 训练模型时添加学习率调度器
# conv_hist = model.fit(
#     x_train_mixed, y_train_mixed,
#     batch_size=128, epochs=30,
#     validation_data=(x_test_mixed, y_test_mixed),
#     callbacks=[lr_callback]
# )

print("第二种")
# # 第二种
# # 连续几个周期损失没有改善时降低学习率
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau
# # 编译模型
# model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"]) # 设定起始学习率
#
# # 创建ReduceLROnPlateau回调函数
# reduce_lr_callback = ReduceLROnPlateau(
#     monitor='val_loss',  # 监控指标：验证集损失
#     factor=0.5,  # 学习率降低的因子，可以根据需要调整
#     patience=3,  # 连续几个周期损失没有改善时降低学习率
#     min_lr=1e-6,  # 学习率下限
#     verbose=1  # 显示学习率变化信息
# )
#
# # 训练模型时添加ReduceLROnPlateau回调函数
# history = model.fit(
#     x_train_mixed, y_train_mixed,
#     batch_size=128,
#     epochs=30,
#     validation_data=(x_test_mixed, y_test_mixed),
#     callbacks=[reduce_lr_callback]
# )

print("第三种")
# # 第三种
# # 每五个周期降低一次学习率
# from tensorflow.keras.optimizers import Adam
# # 自定义学习率调度器
# class CustomLearningRateScheduler(keras.callbacks.Callback):
#     def __init__(self, initial_lr, schedule):
#         super().__init__()
#         self.initial_lr = initial_lr
#         self.schedule = schedule
#
#     def on_epoch_begin(self, epoch, logs=None):
#         # 计算当前学习率
#         lr = self.schedule(epoch, self.initial_lr)
#         # 设置当前学习率到优化器
#         tf.keras.backend.set_value(self.model.optimizer.lr, lr)
#         print(f"\nEpoch {epoch + 1}: Learning rate is {lr:.6f}")
#
# # 定义学习率调度函数
# def lr_schedule(epoch, initial_lr):
#     # 根据需求定义学习率调度规则
#     if epoch < 5:
#         return initial_lr
#     elif epoch < 10:
#         return initial_lr * 0.1
#     elif epoch < 15:
#         return initial_lr * 0.05
#     elif epoch < 20:
#         return initial_lr * 0.01
#     elif epoch < 25:
#         return initial_lr * 0.005
#     else:
#         return initial_lr * 0.001
#
#
# # 编译模型
# model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])
#
# # 初始化自定义学习率调度器
# initial_lr = 0.01  # 设置初始学习率
# lr_scheduler = CustomLearningRateScheduler(initial_lr, lr_schedule)
#
# # 模型训练
# conv_hist = model.fit(
#     x_train_mixed, y_train_mixed,
#     batch_size=128, epochs=30,  # 根据你的需求设置训练周期
#     validation_data=(x_test_mixed, y_test_mixed),
#     callbacks=[lr_scheduler],  # 添加自定义学习率调度器回调
# )

print("第四种")
# 第四种
# expontential decay(指数性衰减）
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import Callback
#
# class LearningRateCallback(Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         lr = float(self.model.optimizer.learning_rate(self.model.optimizer.iterations))
#         print(f'Learning rate for epoch {epoch + 1} is: {lr}')
#
#
# exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
#                         initial_learning_rate=2e-2, decay_steps=1000, decay_rate=0.95)
#
# model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=exponential_decay), metrics=["accuracy"])
#
# lr_callback = LearningRateCallback()
#
# conv_hist = model.fit(
#     x_train_mixed, y_train_mixed,
#     batch_size=256,
#     epochs=50,
#     validation_data=(x_test_mixed, y_test_mixed),
#     callbacks=[lr_callback]
# )

print("第五种")

# 第五种
# polynomial decay （多项式衰减学习率）
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import Callback
#
# class LearningRateCallback(Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         lr = float(self.model.optimizer.learning_rate(self.model.optimizer.iterations))
#         print(f'Learning rate for epoch {epoch + 1} is: {lr}')
#
#
# # 编译模型
# polynomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate =0.02,decay_steps=10000,
#                                                                  end_learning_rate=0.001,power=2,cycle=True,name=None)
# model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=polynomial_decay), metrics=["accuracy"])
#
# lr_callback = LearningRateCallback()
#
# conv_hist = model.fit(
#     x_train_mixed, y_train_mixed,
#     batch_size=256,
#     epochs=50,
#     validation_data=(x_test_mixed, y_test_mixed),
#     callbacks=[lr_callback]
# )

print("第六种")
# 第六种
# inverse time decay(逆时衰减学习率）
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import Callback
#
# class LearningRateCallback(Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         lr = float(self.model.optimizer.learning_rate(self.model.optimizer.iterations))
#         print(f'Learning rate for epoch {epoch + 1} is: {lr}')
#
#
# inverse_time_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
#                      initial_learning_rate=0.02, decay_steps=8000, decay_rate=0.95, staircase=False)
#
# model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=inverse_time_decay), metrics=["accuracy"])
#
# lr_callback = LearningRateCallback()
#
# conv_hist = model.fit(
#     x_train_mixed, y_train_mixed,
#     batch_size=256,
#     epochs=50,
#     validation_data=(x_test_mixed, y_test_mixed),
#     callbacks=[checkpoint,lr_callback]
# )

print("第七种")
# 第七种
# cosine decay（余弦衰减学习率）
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import Callback
#
# class LearningRateCallback(Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         lr = float(self.model.optimizer.learning_rate(self.model.optimizer.iterations))
#         print(f'Learning rate for epoch {epoch + 1} is: {lr}')
#         self.model.learning_rates = getattr(self.model, 'learning_rates', [])
#         self.model.learning_rates.append(lr)
#
#
# cosine_decay = tf.keras.experimental.CosineDecay(
#                 initial_learning_rate=0.02, decay_steps=8000,alpha=0.0,name=None,warmup_target=None,warmup_steps=0)
#
# model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=cosine_decay), metrics=["accuracy"])
#
# lr_callback = LearningRateCallback()
#
# conv_hist = model.fit(
#     x_train_mixed, y_train_mixed,
#     batch_size=256,
#     epochs=50,
#     validation_data=(x_test_mixed, y_test_mixed),
#     callbacks=[checkpoint,lr_callback]
# )
#
# plt.plot(range(1, len(lr_callback.model.learning_rates) + 1), lr_callback.model.learning_rates,label="cosine decay")
# plt.xlabel('Epoch')
# plt.ylabel('Learning Rate')
# plt.title('Learning Rate Schedule')
# plt.legend()
# plt.grid()
# plt.show()