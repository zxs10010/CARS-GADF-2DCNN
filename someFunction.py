import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

################################################################################
# 从桌面上的mat文件中读取数据，然后将数据处理成均值为0，标准差为1的形式
def getdata_corn():
    dataFile = 'C://Users//16682//Desktop//test//'
    dataName = 'corn.mat'
    data = scio.loadmat(dataFile + dataName)
    # scio.loadmat用法： '''https://blog.csdn.net/weixin_44871341/article/details/106457765'''
    datastr5 = data['m5spec']
    datastr5p = data['mp5spec']
    datastr6p = data['mp6spec']
    dataAll = data['propvals']
    # print(datastr5[0,0]['data'])         #成功
    # https://blog.csdn.net/Stanford6/article/details/115623698
    # https://blog.csdn.net/qq_35667901/article/details/103937393
    data5 = datastr5[0, 0]['data']  # 读取mat文件结构体内的数据(如上网站)
    datap5 = datastr5p[0, 0]['data']
    datap6 = datastr6p[0, 0]['data']
    dataContAll = dataAll[0, 0]['data']
    moisture = np.empty(80, dtype=float)
    changdu = 0
    # print(data5.mean(axis=0))
    # print(data5.mean(axis=0).shape)
    # print(data5.std(axis=0))
    # print("######################################################")
    # print(dataContAll.shape)
    # print(type(dataContAll))
    # print(dataContAll)
    # print("######################################################")
    for i in dataContAll:  # TODO 此处重写方便返回四种成分
        moisture[changdu] = i[1]
        changdu = changdu + 1
    '''data5_mean = data5 - np.mean(data5, axis=0)  # 对数据进行零中心化(重要)
    cov = np.dot(data5_mean.T, data5_mean) / data5_mean.shape[0]  # 得到数据的协方差矩阵
    U, S, V = np.linalg.svd(cov)
    data5 = np.dot(data5_mean, U)  # 对数据去相关性
    data5 = np.dot(data5_mean, U[:, :640])  # Xrot_reduced 变成 [N x 100]'''
    '''pca = PCA(n_components=80)
    pca.fit(data5)
    data5=pca.transform(data5)
    print(data5.shape)'''
    '''axis 不设置值，对 m*n 个数求均值，返回一个实数
        axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
        axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵'''

    '''def awgn(x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / len(x)
        npower = xpower / snr
        noise = np.random.randn(len(x)) * np.sqrt(npower)
        return noise + x

    def data_add_corn(db):
        dB = db
        data5_add = np.array([awgn(data5[0], dB)])
        for i in range(1, 80):
            n = np.array([awgn(data5[i], dB)])
            data5_add = np.append(data5_add, n, axis=0)
        n = data5_add - data5
        return data5_add

    data5_1 = data_add_corn(50)
    data5_2 = data_add_corn(60)
    # data5_3 = data_add_corn(70)
    # data5_4 = data_add_corn(80)
    # print(moisture)
    data5 = np.append(data5, data5_1, axis=0)
    moisture_1 = np.append(moisture, moisture)
    data5 = np.append(data5, data5_2, axis=0)
    moisture = np.append(moisture_1, moisture)
    # data5 = np.append(data5, data5_3, axis=0)
    # moisture_3 = np.append(moisture_2, moisture)
    # data5 = np.append(data5, data5_4, axis=0)
    # moisture = np.append(moisture_3, moisture)'''

    data5 -= data5.mean(axis=0)  # 将数据压缩到0-1之间
    data5 /= data5.std(axis=0)  # std()求标准差
    datap5 -= datap5.mean(axis=0)
    datap5 /= datap5.std(axis=0)
    datap6 -= datap6.mean(axis=0)
    datap6 /= datap6.std(axis=0)
    return moisture, data5, datap5, datap6


def drawAcc(history):
    ########################################################################################################
    # 绘制训练精度以及验证精度
    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


########################################################################################################
def drawLoss(history):
    # 绘制训练损失和验证损失
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_valuse = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Train loss')
    plt.plot(epochs, val_loss_valuse, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


#########################################################################################################
# 返回测试数据
def getTestData_corn(chengfenshui, d5, dp5, dp6):
    '''idx = np.argsort(chengfenshui)
    moisture = chengfenshui[idx]
    d5 = d5[idx]

    x = 2
    y = d5[x]
    z = moisture[x]
    moisture_empty = np.array(z)
    d5_empty = np.array([y])

    index = np.array([6, 9, 15, 16, 20, 27, 28, 35, 36, 40, 46, 48, 52, 56, 62, 64, 71, 74, 77])
    for i in index:
        y = np.array([d5[i]])
        z = moisture[i]
        moisture_empty = np.append(moisture_empty, z)
        d5_empty = np.append(d5_empty, y, axis=0)
    index = np.append(x, index)

    test_data = d5_empty
    train_data = np.delete(d5, index, axis=0)
    test_lable = moisture_empty
    train_lable = np.delete(moisture, index)'''

    train_data = d5[:60]
    test_data = d5[60:80]
    train_lable = chengfenshui[0:60]
    test_lable = chengfenshui[60:80]

    print(train_data.shape, test_data.shape, train_lable.shape, test_lable.shape)
    print(train_lable.mean(), train_lable.std(), test_lable.mean(), test_lable.std())

    '''# for i in [80, 160, 240, 320]:
    for i in [80, 160]:
        train_data_1 = d5[i:i + 60]
        train_data = np.append(train_data, train_data_1, axis=0)
        test_data_1 = d5[i + 60:i + 80]
        test_data = np.append(test_data, test_data_1, axis=0)
        train_lable_1 = chengfenshui[i:i + 60]
        train_lable = np.append(train_lable, train_lable_1)
        test_lable_1 = chengfenshui[i + 60:i + 80]
        test_lable = np.append(test_lable, test_lable_1)'''

    '''train_data_mean = train_data.mean(axis=0)
    train_data_std = train_data.std(axis=0)
    train_data -= train_data_mean  # 将数据转变为均值为0
    train_data /= train_data_std  # std()求标准差
    test_data -= train_data_mean
    test_data /= train_data_std'''

    train_data = train_data.astype('float64')
    test_data = test_data.astype('float64')
    train_lable = train_lable.astype('float64')
    test_lable = test_lable.astype('float64')
    train_data = train_data.reshape(60, 700, 1)
    test_data = test_data.reshape(20, 700, 1)
    '''train_data = train_data.reshape(60, 80, 1)
    test_data = test_data.reshape(20, 80, 1)'''
    return test_data, test_lable, train_data, train_lable


###########################################################################################################
# 计算RMSEC、RMSEP
def calculate_RMSE(p_value, r_value):
    cal_tem = 0
    cc_len = len(p_value)
    for i in range(cc_len):
        tem = math.pow((p_value[i] - r_value[i]), 2)
        cal_tem = cal_tem + tem
    cal_cc = cal_tem / cc_len
    cal_fin = math.sqrt(cal_cc)
    return cal_fin


##########################################################################################################
# 计算决定系数R^2
def calculate_R2(p_value, r_value):
    # print('-----------------')
    # print(p_value,r_value)
    average = r_value.mean(axis=0)
    # print("平均值：", average)
    cr_len = len(r_value)
    # print(cr_len)
    car_tem = 0
    cars_tem = 0
    for i in range(cr_len):
        temp_r = math.pow(r_value[i] - average, 2)
        car_tem = car_tem + temp_r
    # print(car_tem)
    for i2 in range(cr_len):
        temp_s = math.pow(p_value[i2] - average, 2)
        cars_tem = cars_tem + temp_s
    # print(cars_tem)
    r_2 = (cars_tem / car_tem)
    # print("cr_len长度：",cr_len)
    # print("真值减平均：",car_tem)
    # print("预测值减平均：",cars_tem)
    # print("################")
    return r_2


#########################################################################################################
# 平滑曲线
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


###############################################################################################################
# 向CSV中写入数据
def write_To_Csv(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('new.csv', mode='a', index=False, header=False)
    return

def write_To_Csv_1(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('new_diesel_total_2D_shixiong_gaosi.csv', mode='a', index=False, header=False)
    return



def write_To_Csv_train_lable(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    # 'C:/Users/16682/Desktop/NIRATSpecNet-main - 副本/数据/
    df.to_csv('train_lable.csv', mode='a', index=False, header=False)
    return


def write_To_Csv_test_lable(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('test_lable.csv', mode='a', index=False, header=False)
    return


def write_To_Csv_train_data(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('train_data.csv', mode='a', index=False, header=False)
    return


def write_To_Csv_test_data(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('test_data.csv', mode='a', index=False, header=False)
    return


#############################################################################################################
def calculate_R21(p_value, r_value):
    # print('-----------------')
    # print(p_value,r_value)
    average = r_value.mean(axis=0)
    # print("平均值：", average)
    cr_len = len(r_value)
    # print(cr_len)
    car_tem = 0
    cars_tem = 0
    for i in range(cr_len):
        temp_r = math.pow(r_value[i] - average, 2)
        car_tem = car_tem + temp_r
    # print(car_tem)
    for i2 in range(cr_len):
        temp_s = math.pow(p_value[i2] - r_value[i2], 2)
        cars_tem = cars_tem + temp_s
    # print(cars_tem)
    r_2 = 1 - (cars_tem / car_tem)
    # print("cr_len长度：",cr_len)
    # print("真值减平均：",car_tem)
    # print("预测值减平均：",cars_tem)
    # print("################")
    return r_2


###################################################################################################################
# 返回测试集、验证集
def getTestData_chaiyou(spec, prop):
    train_data = spec[:300]
    test_data = spec[300:]
    train_lable = prop[:300]
    test_lable = prop[300:]
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_lable = train_lable.astype('float32')
    test_lable = test_lable.astype('float32')
    train_data = train_data.reshape(300, 401, 1)
    test_data = test_data.reshape(81, 401, 1)
    return test_data, test_lable, train_data, train_lable


###################################################################################################################
# 返回药片的测试集，验证集
def getTestDataPills(spec, hardness):
    train_data = spec[:400]
    test_data = spec[400:]
    train_lable = hardness[:400]
    test_lable = hardness[400:]
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_lable = train_lable.astype('float32')
    test_lable = test_lable.astype('float32')
    train_data = train_data.reshape(400, 650, 1)
    test_data = test_data.reshape(60, 650, 1)
    return test_data, test_lable, train_data, train_lable


###################################################################################################################
# 返回牛奶的测试集，验证集
def getTestDataMilk(spec, dataprop):
    train_data = spec[7:]
    test_data = spec[:7]
    train_lable = dataprop[7:]
    test_lable = dataprop[:7]
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_lable = train_lable.astype('float32')
    test_lable = test_lable.astype('float32')
    train_data = train_data.reshape(60, 1557, 1)
    test_data = test_data.reshape(7, 1557, 1)
    return test_data, test_lable, train_data, train_lable


###################################################################################################################

# 自定义度量函数
def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


###################################################################################################################

# 从桌面上的mat文件中读取数据，然后将数据处理成均值为0，标准差为1的形式
def getdata_chaiyou():
    dataFile = 'C://Users//16682//Desktop//test//'
    dataName = 'chaiyou.mat'
    data = scio.loadmat(dataFile + dataName)
    # scio.loadmat用法： '''https://blog.csdn.net/weixin_44871341/article/details/106457765'''
    # print(data.keys())
    diesel_spec = data['diesel_spec']
    diesel_prop = data['diesel_prop']
    # print(datastr5[0,0]['data'])         #成功
    # https://blog.csdn.net/Stanford6/article/details/115623698
    # https://blog.csdn.net/qq_35667901/article/details/103937393
    diesel_spec_0 = diesel_spec[0, 0]['data']  # 读取mat文件结构体内的数据(如上网站)
    diesel_prop = diesel_prop[0, 0]['data']

    moisture = np.empty(784, dtype=float)
    changdu = 0
    for i in diesel_prop:  # TODO 此处重写方便返回四种成分
        moisture[changdu] = i[1]
        changdu = changdu + 1
    # print(diesel_spec.shape)
    # print(diesel_spec)
    # print(diesel_spec_0.shape)
    i = moisture > 10
    x = 0
    diesel_prop = np.empty(381, dtype=float)
    diesel_spec = np.empty((381, 401), dtype=float)
    # print(diesel_spec_0[0])
    for a in range(0, 784):
        if i[a]:
            diesel_spec[x] = diesel_spec_0[a]
            diesel_prop[x] = moisture[a]
            x = x + 1

    # diesel_prop=diesel_prop.reshape(7,784)

    # print(data5.mean(axis=0))
    # print(data5.mean(axis=0).shape)
    # print(data5.std(axis=0))
    # print("######################################################")
    # print(dataContAll.shape)
    # print(type(dataContAll))
    # print(dataContAll)
    # print("######################################################")

    diesel_spec -= diesel_spec.mean(axis=0)  # 将数据转变为均值为0
    diesel_spec /= diesel_spec.std(axis=0)  # std()求标准差
    return diesel_spec, diesel_prop


# 从桌面上的mat文件中读取数据，然后将数据处理成均值为0，标准差为1的形式
def getdata_beer():
    dataFile = 'C://Users//16682//Desktop//test//'
    dataName = 'getdata_beer_cars.mat'
    data = scio.loadmat(dataFile + dataName)
    # scio.loadmat用法： '''https://blog.csdn.net/weixin_44871341/article/details/106457765'''
    # print(data.keys())
    xtrain = data['XXtrain']
    xtest = data['XXtest']
    ytrain = data['Ytrain']
    ytest = data['Ytest']

    x = np.append(xtrain, xtest, axis=0)
    y = np.append(ytrain, ytest, axis=0)
    # min_max_scaler = MinMaxScaler()
    # x = min_max_scaler.fit_transform(x)

    print(x.shape, y.shape)

    # xtrain, xtest, ytrain, ytest = ks(x, y)
    print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
    print(ytrain.mean(), ytrain.std(), ytest.mean(), ytest.std())

    # xtrain -= xtrain.mean(axis=0)  # 将数据转变为均值为0
    # xtrain /= xtrain.std(axis=0)  # std()求标准差
    # xtest -= xtest.mean(axis=0)  # 将数据转变为均值为0
    # xtest /= xtest.std(axis=0)  # std()求标准差
    xtrain = xtrain.reshape(40, 576, 1)
    xtest = xtest.reshape(20, 576, 1)
    ytrain = ytrain.reshape(40, )
    ytest = ytest.reshape(20, )
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    ytrain = ytrain.astype('float32')
    ytrain = ytrain.astype('float32')
    return xtrain, ytrain, xtest, ytest


def spxy(x, y, test_size=0.25):
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size
    :return: spec_train :(n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    x_backup = x
    y_backup = y
    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    y = (y - np.mean(y)) / np.std(y)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    for i in range(M - 1):
        xa = x[i, :]
        ya = y[i]
        for j in range((i + 1), M):
            xb = x[j, :]
            yb = y[j]
            D[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)

    Dmax = np.max(D)
    Dymax = np.max(Dy)
    D = D / Dmax + Dy / Dymax

    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)
    index_column = maxD.argmax()

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)

    spec_train = x[m, :]
    target_train = y_backup[m]
    spec_test = x[m_complement, :]
    target_test = y_backup[m_complement]

    return spec_train, spec_test, target_train, target_test


def getTestData_corn_spxy(chengfenshui, d5, dp5, dp6):
    '''idx = np.argsort(chengfenshui)
    moisture = chengfenshui[idx]
    d5 = d5[idx]

    x = 2
    y = d5[x]
    z = moisture[x]
    moisture_empty = np.array(z)
    d5_empty = np.array([y])

    index = np.array([6, 9, 15, 16, 20, 27, 28, 35, 36, 40, 46, 48, 52, 56, 62, 64, 71, 74, 77])
    for i in index:
        y = np.array([d5[i]])
        z = moisture[i]
        moisture_empty = np.append(moisture_empty, z)
        d5_empty = np.append(d5_empty, y, axis=0)
    index = np.append(x, index)

    test_data = d5_empty
    train_data = np.delete(d5, index, axis=0)
    test_lable = moisture_empty
    train_lable = np.delete(moisture, index)'''
    train_data, test_data, train_lable, test_lable = spxy(d5, chengfenshui)
    print(train_data.shape, test_data.shape, train_lable.shape, test_lable.shape)
    print(train_lable.mean(), train_lable.std(), test_lable.mean(), test_lable.std())
    '''train_data_mean = train_data.mean(axis=0)
    train_data_std = train_data.std(axis=0)
    train_data -= train_data_mean  # 将数据转变为均值为0
    train_data /= train_data_std  # std()求标准差
    test_data -= train_data_mean
    test_data /= train_data_std'''

    train_data = train_data.astype('float64')
    test_data = test_data.astype('float64')
    train_lable = train_lable.astype('float64')
    test_lable = test_lable.astype('float64')
    train_data = train_data.reshape(60, 700, 1)
    test_data = test_data.reshape(20, 700, 1)
    '''train_data = train_data.reshape(60, 80, 1)
    test_data = test_data.reshape(20, 80, 1)'''
    return test_data, test_lable, train_data, train_lable





def getTestDate_Grain():
    # dataFile = '../data/'
    dataFile = 'C://Users//16682//Desktop//test//'
    dataName = 'grain_casein.mat'
    data = scio.loadmat(dataFile + dataName)
    datax_train = data['Xcal']
    datax_test = data['Xtest']
    datay_train = data['Ycal']
    datay_test = data['Ytest']
    a = len(datax_train)
    b = len(datax_test)
    # datax_train -= datax_train.mean(axis=0)
    # datax_test -= datax_test.mean(axis=0)
    # datax_train /= datax_train.std(axis=0)
    # datax_test /= datax_test.std(axis=0)
    datax_train = datax_train.astype('float32')
    datax_test = datax_test.astype('float32')
    datay_train = datay_train.astype('float32')
    datay_test = datay_test.astype('float32')
    datax_train = datax_train.reshape(a, 117, 1)
    datax_test = datax_test.reshape(b, 117, 1)
    print(datax_train.shape, datax_test.shape, datay_train.shape, datay_test.shape)

    return datax_test, datay_test, datax_train, datay_train


def getTestData_corn_min(chengfenshui, d5, dp5, dp6):
    idx = np.argsort(chengfenshui)
    moisture = chengfenshui[idx]
    d5 = d5[idx]

    '''x = 2
    y = d5[x]
    z = moisture[x]
    moisture_empty = np.array(z)
    d5_empty = np.array([y])

    index = np.array([6, 9, 15, 16, 20, 27, 28, 35, 36, 40, 46, 48, 52, 56, 62, 64, 71, 74, 77])
    for i in index:
        y = np.array([d5[i]])
        z = moisture[i]
        moisture_empty = np.append(moisture_empty, z)
        d5_empty = np.append(d5_empty, y, axis=0)
    index = np.append(x, index)

    test_data = d5_empty
    train_data = np.delete(d5, index, axis=0)
    test_lable = moisture_empty
    train_lable = np.delete(moisture, index)'''

    train_data_1 = d5[:30]
    train_data_2=d5[50:80]
    train_data=np.append(train_data_1,train_data_2,axis=0)
    test_data = d5[30:50]

    train_lable_1 = moisture[0:30]
    train_lable_2 = moisture[50:80]
    train_lable=np.append(train_lable_1,train_lable_2,axis=0)
    test_lable = moisture[30:50]

    print(train_data.shape, test_data.shape, train_lable.shape, test_lable.shape)
    print(train_lable.mean(), train_lable.std(), test_lable.mean(), test_lable.std())

    '''# for i in [80, 160, 240, 320]:
    for i in [80, 160]:
        train_data_1 = d5[i:i + 60]
        train_data = np.append(train_data, train_data_1, axis=0)
        test_data_1 = d5[i + 60:i + 80]
        test_data = np.append(test_data, test_data_1, axis=0)
        train_lable_1 = chengfenshui[i:i + 60]
        train_lable = np.append(train_lable, train_lable_1)
        test_lable_1 = chengfenshui[i + 60:i + 80]
        test_lable = np.append(test_lable, test_lable_1)'''

    '''train_data_mean = train_data.mean(axis=0)
    train_data_std = train_data.std(axis=0)
    train_data -= train_data_mean  # 将数据转变为均值为0
    train_data /= train_data_std  # std()求标准差
    test_data -= train_data_mean
    test_data /= train_data_std'''

    train_data = train_data.astype('float64')
    test_data = test_data.astype('float64')
    train_lable = train_lable.astype('float64')
    test_lable = test_lable.astype('float64')
    train_data = train_data.reshape(60, 700, 1)
    test_data = test_data.reshape(20, 700, 1)
    '''train_data = train_data.reshape(60, 80, 1)
    test_data = test_data.reshape(20, 80, 1)'''
    return test_data, test_lable, train_data, train_lable
