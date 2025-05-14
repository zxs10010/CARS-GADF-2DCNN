import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyts.image import MarkovTransitionField, GramianAngularField
from Getdata_grain_casein import getdata_Grain_casein
# from Getdata_grain_casein_pill import getdata_Grain_casein
# from Getdata_grain_casein_lactate import getdata_Grain_casein
import scipy.io as sio  # 导入scipy.io用于保存.mat文件

# 设置字体
plt.rc('font', family='Times New Roman')


# 函数用于生成数据并保存为.mat文件
def generate_and_save_data(select_method, cc, filename):
    test_data, test_label, train_data, train_label, n_components, a, b = getdata_Grain_casein(fs=select_method, c=cc)
    train_data = train_data.reshape(a, 1, n_components)
    train_data0 = train_data[0]

    mtf = GramianAngularField(image_size=n_components, method='difference')
    train_data_mtf = mtf.fit_transform(train_data0)

    # 保存数据为.mat文件
    sio.savemat(filename, {'train_data_mtf': train_data_mtf})


# 保存四个不同的训练数据
generate_and_save_data('crs', 6, 'data_crs.mat')
generate_and_save_data('cars', 11, 'data_cars.mat')
generate_and_save_data('iriv', 19, 'data_iriv.mat')
generate_and_save_data('vcpa', 3, 'data_vcpa.mat')
