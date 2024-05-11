import numpy as np
import os
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.utils import to_categorical
from pretty_midi import PrettyMIDI
import re



# MIDI文件到钢琴卷帘图像的转换
def midi_to_piano_roll(midi_path, num_pitches=88, fs=24):
    midi_data = PrettyMIDI(midi_path)
    piano_roll = np.zeros((num_pitches, fs))

    for instrument in midi_data.instruments:
        # 检查乐器是否为钢琴（乐器编号为1）
        if instrument.program == 1:  # 钢琴的MIDI程序编号通常为1
            for note in instrument.notes:
                start_time = int(note.start * fs)
                end_time = int(note.end * fs)
                pitch = note.pitch
                # 确保音符的音高在钢琴卷帘图的范围内
                if 0 <= pitch < num_pitches:
                    piano_roll[pitch, start_time:end_time] = 1

    return piano_roll

# 将钢琴卷帘图像转化为CNN输入格式
def prepare_data(midi_files_path, num_pitches=88, fs=24, num_classes=128):
    X = []  # 用于存储钢琴卷帘图
    y = []  # 用于存储标签

    for midi_file in os.listdir(midi_files_path):
        if midi_file.endswith('.midi'):
            midi_path = os.path.join(midi_files_path, midi_file)
            piano_roll = midi_to_piano_roll(midi_path, num_pitches, fs)
            X.append(piano_roll)

            # 从文件名中提取数字部分作为标签
            label = int(re.search(r'\d+', midi_file).group())
            y.append(label % num_classes)  # 确保标签在0到num_classes-1之间

    X = np.array(X)
    y = to_categorical(y, num_classes=num_classes)

    # 形状调整以匹配CNN的输入要求
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    return X, y

# 构建CNN模型
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




# 主程序
if __name__ == '__main__':
    midi_files_path = r"D:\A下载\maestro-v3.0.0-midi\train\maestro-v3.0.0\2004" # MIDI文件所在目录
    num_pitches = 88  # 钢琴的音高数量
    fs = 24  # 时间分辨率
    num_classes = 128  # 类别数量，根据数据集确定




    # 准备数据
    X, y = prepare_data(midi_files_path, num_pitches, fs, num_classes)

    # 划分数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2], 1)  # 确保输入形状匹配模型
    model = build_model(input_shape, num_classes)

    # 训练模型
    batch_size = 32
    epochs = 10
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

    # 评估模型
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # 可视化训练过程中的准确率和损失
    plt.figure(figsize=(12, 4))

    # 绘制准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    # 绘制损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epoch')

    # 显示图表
    plt.show()

