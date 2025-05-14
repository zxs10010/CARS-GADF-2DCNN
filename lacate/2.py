import pandas as pd
import scipy.io as scio
import numpy as np

import CARS
from Write_to_csv import write_RMSECVmin


def getdata_beer_cars():
    dataFile = 'C://Users//16682//Desktop//test//'
    dataName = 'grain_lactate.mat'
    data = scio.loadmat(dataFile + dataName)
    # print(data)
    datax_train = data['Xcal']
    datax_test = data['Xtest']
    datay_train = data['Ycal']
    datay_test = data['Ytest']
    datax_train = datax_train.reshape(153, 117)
    datax_test = datax_test.reshape(78, 117)

    print(datax_train.shape, datax_test.shape, datay_train.shape, datay_test.shape)

    datax = np.append(datax_train, datax_test, axis=0)
    datay = np.append(datay_train, datay_test, axis=0)
    print(datay_train.max() - datay_train.min())
    print(datay_test.max() - datay_test.min())

    minRMSECV = np.array([])
    for _ in range(20):
        lis, RMSECV = CARS.CARS_Cloud(datax, datay)
        # print("获取波段数：",len(lis))
        # print(lis)
        l = len(lis)
        RMSECVmin = min(RMSECV)
        minRMSECV = np.append(minRMSECV, RMSECVmin)
        # print(min(RMSECV))

        write_data = [(lis, l,RMSECVmin)]
        write_RMSECVmin(write_data)

    lis = np.argmin(minRMSECV)
    minR = min(minRMSECV)
    write_data = [(lis, minR)]
    write_RMSECVmin(write_data)
    # X_ = X[:,lis]
    # print(X_.shape)


if __name__ == '__main__':
    getdata_beer_cars()
