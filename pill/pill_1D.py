import os
# import pandas as pd
# from keras import metrics
# import keras
import tensorflow as tf
import keras
from keras.layers import Conv2D, BatchNormalization, Activation, Add
from keras.layers import Input, MaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.layers import Dense, Dropout, Softmax
from keras.models import Model
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
from keras import layers
from keras import models
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam, Adamax, Nadam
import someFunction as sF
from keras.models import load_model
import datetime
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField, GramianAngularField
import numpy as np
from keras import initializers
# tf.keras.layers
from Getdata_grain_casein import getdata_Grain_casein
from Write_to_csv import write_to_csv_2D_diff,write_to_csv_2D_sum,write_to_csv_1D

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def grain_casein_1D(aa,cc,i, v, f, s):
    now = datetime.datetime.now()
    now_s = now.strftime("%Y-%m-%d-%H-%M-%S")
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    '''config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)'''
    # 准备几个参数，用于后续的自动化
    epochs_au = 400
    batch_size_au = 4
    select_method = 'cars'
    if aa < 1:
        select_method = 'cars'
    elif aa < 2:
        select_method = 'iriv'
    elif aa < 3:
        select_method = 'vcpa'
    if s < 1:
        kernel_initializer = 'HeNormal'
    else:
        kernel_initializer = 'glorot_normal'

    kernel_size = f
    if v < 1:
        jihuo = 'relu'
    elif v < 2:
        jihuo = 'tanh'
    elif v < 3:
        jihuo = 'sigmoid'
    image_size = 117
    n_bins = 8
    '''if i < 5:
        optimiz = 'SGD'
    elif i < 10:
        optimiz = 'Adagrad'
    elif i < 15:
        optimiz = 'Adadelta'
    elif i < 20:
        optimiz = 'RMSprop'
    elif i < 25:
        optimiz = 'Adam'
    elif i < 30:
        optimiz = 'Adamax'
    elif i < 35:
        optimiz = 'Nadam'''
    if i < 100:
        adam = Adam(learning_rate=0.001)
        optimiz = adam
    # elif i < 10:
    #     adamax = Adamax(learning_rate=0.00005)
    #     optimiz = adamax
    # elif i < 15:
    #     nadam = Nadam(learning_rate=0.00005)
    #     optimiz = nadam

    # callback用法:'''https://blog.csdn.net/wf592523813/article/details/95027246?ops_request_misc'''
    callback_list_test = \
        [
            ModelCheckpoint(filepath='E://NIRATSpecNet-main//grain//grain_casein//h5_1D//'+now_s + '.h5',  # 文件路径 存在当前路径下吧 还好找
                            monitor='val_loss',  # 监控指标 检测测试集损失值
                            save_best_only=True,  # 只保持最佳模型
                            # save_weights_only=False
                            )
        ]

    test_data, test_lable, train_data, train_lable,n_components ,a,b= getdata_Grain_casein(fs='select_method',c=cc)
    print(test_data.shape)

    input = layers.Input(shape=(n_components, 1), dtype='float32')
    net = layers.Conv1D(16, 3, activation=jihuo, padding='same',
                        kernel_initializer=kernel_initializer,)(input)

    b_1 = layers.Conv1D(16, 7, activation=jihuo, padding='same',
                        kernel_initializer=kernel_initializer,)(net)
    b_1 = layers.Conv1D(16, 3, activation=jihuo, padding='same',
                        kernel_initializer=kernel_initializer,)(b_1)
    b_1 = layers.Conv1D(8, 3, activation=jihuo, padding='same',
                        kernel_initializer=kernel_initializer,)(b_1)

    b_2 = layers.Conv1D(16, 5, activation=jihuo, padding='same',
                        kernel_initializer=kernel_initializer,)(net)
    b_2 = layers.Conv1D(16, 3, activation=jihuo, padding='same',
                        kernel_initializer=kernel_initializer,)(b_2)
    b_2 = layers.Conv1D(8, 3, activation=jihuo, padding='same',
                        kernel_initializer=kernel_initializer,)(b_2)

    # b_3 = layers.Conv2D(16, 7, activation=jihuo, padding='same')(net)
    # b_3 = layers.Conv2D(16, 5, activation=jihuo, padding='same')(b_3)
    # b_3 = layers.Conv2D(8, 3, activation=jihuo, padding='same')(b_3)

    # b_4 = layers.AvgPool2D(1)(net)
    # b_4 = layers.Conv2D(16, 7, activation=jihuo, padding='same')(b_4)
    # b_4 = layers.Conv2D(16, 5, activation=jihuo, padding='same')(b_4)
    # b_4 = layers.Conv2D(8, 3, activation=jihuo, padding='same')(b_4)
    #
    # b_5 = layers.AvgPool2D(1)(net)
    # b_5 = layers.Conv2D(16, 5, activation=jihuo, padding='same')(b_5)
    # b_5 = layers.Conv2D(8, 3, activation=jihuo, padding='same')(b_5)
    # b_5 = layers.Conv2D(8, 3, activation=jihuo, padding='same')(b_5)

    net = layers.concatenate([b_1, b_2, ], axis=1)
    output = layers.Flatten()(net)
    output = layers.Dense(4)(output)
    output = layers.Dense(4)(output)
    output = layers.Dense(2)(output)
    output = layers.Dense(1)(output)
    model = Model(input, output)

    # model.summary()  # 输出模型各层的参数状况（有多少个参数）
    model.compile(
        optimizer=RMSprop(),
        # optimizer=optimiz,
        loss='mse')
    history = model.fit(train_data, train_lable,
                        epochs=epochs_au,
                        batch_size=batch_size_au,
                        validation_data=(test_data, test_lable),
                        callbacks=callback_list_test
                        )

    # sF.drawLoss(history)  # 绘制当前的验证曲线
    model = load_model('E://NIRATSpecNet-main//grain//grain_casein//h5_1D//'+now_s + '.h5',)
    # model = load_model('model.h5', custom_objects={'GroupNormalization': GroupNormalization()})
    result_trian = model.predict(train_data)
    result_predict = model.predict(test_data)
    rmsec = sF.calculate_RMSE(result_trian, train_lable)  # 训练集上的RMSE
    rmsep = sF.calculate_RMSE(result_predict, test_lable)  # 测试集上的RMSE
    r_2_t = sF.calculate_R21(result_trian, train_lable)  # 训练集上的R_2
    r2_t = sF.calculate_R2(result_trian, train_lable)
    # print(rmsec)
    print(sF.calculate_R21(result_trian, train_lable))
    print(sF.calculate_R2(result_trian, train_lable))
    r_2_p = sF.calculate_R21(result_predict, test_lable)  # 测试集上得R_2
    r2_p = sF.calculate_R2(result_predict, test_lable)
    # print(rmsep)
    print(sF.calculate_R21(result_predict, test_lable))
    print(sF.calculate_R2(result_predict, test_lable))
    '''print(result_trian)
    print(result_predict)
    print("Root Mean Square Error of Calibrationh is : %g" % rmsec)
    print("训练集上得决定系数：%f" % r_2_t)
    print("Root Mean Square Error of Prediction is : %g" % rmsep)
    print("测试集上得决定系数：%f" % r_2_p)'''
    # 下面的代码用于自动记录实验数据

    write_data = [
        (now_s, epochs_au, batch_size_au, kernel_size, n_components, n_bins, jihuo, optimiz, kernel_initializer, rmsec,
         r_2_t, r2_t, rmsep, r_2_p, r2_p)]
    write_to_csv_1D(write_data,select_method,cc)
    return rmsep, r_2_p


if __name__ == '__main__':
    for bb in range(0, 1):  # 变量选择方法
        for cc in range(19, 20):  # 20个已选变量
            for t in range(0, 1):  # 激活函数：t
                for s in range(0, 1):  # 权值初始化 s
                    for q in range(0, 20):  # 优化器 q
                        for w in (2,):  # 卷积核大小 w
                            grain_casein_1D(bb,cc,q, t, w, s)
