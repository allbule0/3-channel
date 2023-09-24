# 扩展版块：寻找最佳学习率

# 编译模型：默认情况
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
optimizer = model.optimizer
current_learning_rate = optimizer.learning_rate.numpy()
print("当前学习率:", current_learning_rate)


# 寻找最佳学习率
import numpy as np
import matplotlib.pyplot as plt

# 定义一系列学习率范围
learning_rates = np.logspace(-6, -1, num=20)  # 这里的参数可灵活调整

# 记录损失
losses = []

for lr in learning_rates:
    # 创建并编译模型
    model = mcnndep(inputshape, filterAg, classnum)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])

    # 训练模型，通常使用少量周期
    history = model.fit(x_train_mixed, y_train_mixed, batch_size=128, epochs=5, validation_data=(x_test_mixed, y_test_mixed), verbose=0)

    # 记录最终损失
    final_loss = history.history['val_loss'][-1]
    losses.append(final_loss)

# 绘制学习率和损失的关系图
plt.semilogx(learning_rates, losses)
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.show()

# 选择最佳学习率
best_learning_rate = learning_rates[np.argmin(losses)]
print("最佳学习率:", best_learning_rate)

# 接下来可将得到的最佳学习率应用在mmodel.py文件中指定学习率
