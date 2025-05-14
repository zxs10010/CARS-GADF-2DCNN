import scipy.io as scio
import numpy as np
from sklearn.decomposition import PCA


def getdata_Grain_casein(fs,c):
    n_components = 50
    dataFile = 'C://Users//16682//Desktop//test//'
    dataName = 'pill1.mat'
    dataName_select='pill_select.mat'
    data = scio.loadmat(dataFile + dataName)
    data_select = scio.loadmat(dataFile + dataName_select)
    datax_train = data['Xtrain']
    datax_test = data['Xtest']
    datay_train = data['ytrain']
    datay_test = data['ytest']

    datax = np.append(datax_train, datax_test, axis=0)
    datay = np.append(datay_train, datay_test, axis=0)
    datax_train, datax_test, datay_train, datay_test = ks(datax, datay)
    print(max(datay_train))
    print(max(datay_test))
    print(np.max(datay_train), np.min(datay_train), np.mean(datay_train), np.std(datay_train), np.std(datay_train))
    print(np.max(datay_test), np.min(datay_test), np.mean(datay_test), np.std(datay_test), np.std(datay_test))
    print(datax_train.shape, datax_test.shape, datay_train.shape, datay_test.shape)
    # c=np.array([datay_test[np.argmax(datay_test)]])
    # print(c)
    # datay_train=np.append(datay_train, c, axis=0)
    # datay_test=np.delete(datay_test,np.argmax(datay_test))
    # print(datay_train.shape)
    # print(datay_test.shape)


    datax = np.append(datax_train, datax_test, axis=0)
    datay = np.append(datay_train, datay_test, axis=0)
    # print(datay_train.max() - datay_train.min())
    # print(datay_test.max() - datay_test.min())

    if fs == 'pca':
        pca = PCA(n_components=n_components)  # 把二维的原数据降成一维
        datax = pca.fit_transform(datax)  # 把原数据X传入方法，返回降维后的数据，等价于pca.fit(X) pca.transform(X)
    elif fs == 'cars':
        lis = data_select['pill_cars'][0][c][0]
        lis = lis
        n_components = len(lis)
        # print(n_components)
        datax = datax[:, lis]
    elif fs == 'iriv':
        lis = data_select['pill_iriv'][0][c][0]
        lis = lis - 1
        n_components = len(lis)
        # print(n_components)
        datax = datax[:, lis]
    elif fs == 'vcpa':
        lis = data_select['pill_vcpa'][0][c][0]
        lis = lis - 1
        n_components = len(lis)
        # print(n_components)
        datax = datax[:, lis]
    else:
        n_components = 404
        pass

    datax_train = datax[0:231, :]
    datax_test = datax[231:, :]
    datay_train = datay[0:231]
    datay_test = datay[231:]
    print(np.max(datay_train), np.min(datay_train), np.mean(datay_train), np.std(datay_train), np.std(datay_train))
    print(np.max(datay_test), np.min(datay_test), np.mean(datay_test), np.std(datay_test), np.std(datay_test))
    # datax_train, datax_test, datay_train, datay_test = ks(datax, datay)
    # print(datax_train.shape, datax_test.shape, datay_train.shape, datay_test.shape)
    datax = np.append(datax_train, datax_test, axis=0)
    datay = np.append(datay_train, datay_test, axis=0)
    # print(datay_train.max() - datay_train.min())
    # print(datay_test.max() - datay_test.min())

    a = len(datax_train)
    b = len(datax_test)

    # def awgn(x, snr):
    #     snr = 10 ** (snr / 10.0)
    #     xpower = np.sum(x ** 2) / len(x)
    #     npower = xpower / snr
    #     noise = np.random.randn(len(x)) * np.sqrt(npower)
    #     return noise + x
    #
    # def data_add_corn(db):
    #     dB = db
    #     datax_add = np.array([awgn(datax[0], dB)])
    #     for i in range(1, a + b):
    #         n = np.array([awgn(datax[i], dB)])
    #         datax_add = np.append(datax_add, n, axis=0)
    #     n = datax_add - datax
    #     return datax_add

    # datax_1 = data_add_corn(50)
    # datax_2 = data_add_corn(60)
    # datax_3 = data_add_corn(70)
    # datax_4 = data_add_corn(80)
    #
    # # datax_train = datax[:a]
    # # datax_test = datax[a:]
    # datax_1_train = datax_1[:a]
    # datax_1_test = datax_1[a:]
    # datax_2_train = datax_2[:a]
    # datax_2_test = datax_2[a:]
    # datax_3_train = datax_3[:a]
    # datax_3_test = datax_3[a:]
    # datax_4_train = datax_4[:a]
    # datax_4_test = datax_4[a:]
    #
    # data0 = np.array([datax_train[0]])
    # # print(datax_train.shape)
    # data0_1 = np.array([datax_1_train[0]])
    # # print(datax_1_train.shape)
    # data = np.append(data0, data0_1, axis=0)
    # data0_2 = np.array([datax_2_train[0]])
    # data = np.append(data, data0_2, axis=0)
    # data0_3 = np.array([datax_3_train[0]])
    # data = np.append(data, data0_3, axis=0)
    # data0_4 = np.array([datax_4_train[0]])
    # data = np.append(data, data0_4, axis=0)
    #
    # # print(data.shape)
    # for i in range(1, a):
    #     data0 = np.array([datax_train[i]])
    #     data0_1 = np.array([datax_1_train[i]])
    #     data_ = np.append(data0, data0_1, axis=0)
    #     data0_2 = np.array([datax_2_train[i]])
    #     data_ = np.append(data_, data0_2, axis=0)
    #     data0_3 = np.array([datax_3_train[i]])
    #     data_ = np.append(data_, data0_3, axis=0)
    #     data0_4 = np.array([datax_4_train[i]])
    #     data_ = np.append(data_, data0_4, axis=0)
    #     data = np.append(data, data_, axis=0)
    # datax_train = data
    # # print(datax_train.shape)
    #
    # # print('-------------------------------')
    #
    # data0 = np.array([datax_test[0]])
    # # print(datax_test.shape)
    # data0_1 = np.array([datax_1_test[0]])
    # # print(datax_1_test.shape)
    # data = np.append(data0, data0_1, axis=0)
    # data0_2 = np.array([datax_2_test[0]])
    # data = np.append(data, data0_2, axis=0)
    # data0_3 = np.array([datax_3_test[0]])
    # data = np.append(data, data0_3, axis=0)
    # data0_4 = np.array([datax_4_test[0]])
    # data = np.append(data, data0_4, axis=0)
    #
    # # print(data.shape)
    # for i in range(1, b):
    #     data0 = np.array([datax_test[i]])
    #     data0_1 = np.array([datax_1_test[i]])
    #     data_ = np.append(data0, data0_1, axis=0)
    #     data0_2 = np.array([datax_2_test[i]])
    #     data_ = np.append(data_, data0_2, axis=0)
    #     data0_3 = np.array([datax_3_test[i]])
    #     data_ = np.append(data_, data0_3, axis=0)
    #     data0_4 = np.array([datax_4_test[i]])
    #     data_ = np.append(data_, data0_4, axis=0)
    #     data = np.append(data, data_, axis=0)
    # datax_test = data
    # # print(datax_test.shape)
    #
    # # print('-------------------------------')
    #
    # data0 = np.array(datay_train[0])
    # # print(datay_train.shape)
    # data = np.append(data0, data0, )
    # data = np.append(data, data0, )
    # data = np.append(data, data0, )
    # data = np.append(data, data0, )
    #
    # # print(data.shape)
    # for i in range(1, a):
    #     data0 = np.array(datay_train[i])
    #     data_ = np.append(data0, data0)
    #     data_ = np.append(data_, data0)
    #     data_ = np.append(data_, data0)
    #     data_ = np.append(data_, data0)
    #     data = np.append(data, data_)
    # datay_train = data
    # # print(datay_train.shape)
    #
    # # print('-------------------------------')
    #
    # data0 = np.array(datay_test[0])
    # # print(datay_test.shape)
    # data = np.append(data0, data0, )
    # data = np.append(data, data0, )
    # data = np.append(data, data0, )
    # data = np.append(data, data0, )
    #
    # # print(data.shape)
    # for i in range(1, b):
    #     data0 = np.array(datay_test[i])
    #     data_ = np.append(data0, data0)
    #     data_ = np.append(data_, data0)
    #     data_ = np.append(data_, data0)
    #     data_ = np.append(data_, data0)
    #     data = np.append(data, data_)
    # datay_test = data
    # # print(datay_test.shape)
    # a = 5 * a
    # b = 5 * b

    datax_train = datax_train.astype('float32')
    datax_test = datax_test.astype('float32')
    datay_train = datay_train.astype('float32')
    datay_test = datay_test.astype('float32')
    # print(datay_test)
    # print(data['Ytest'])

    # print(datax_train.shape, datax_test.shape, datay_train.shape, datay_test.shape)
    # print(datay_test)
    return datax_test, datay_test, datax_train, datay_train, n_components, a, b





if __name__ == '__main__':
    getdata_Grain_casein(fs='vcp',c=1)
