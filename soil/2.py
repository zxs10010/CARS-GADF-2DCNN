import pandas as pd
import scipy.io as scio
import numpy as np

import CARS
from Write_to_csv import write_RMSECVmin


def getdata_beer_cars():
    n_components = 117
    dataFile = 'C://Users//16682//Desktop//test//'
    dataName = 'soil.mat'
    data = scio.loadmat(dataFile + dataName)
    datax_train = data['Xtrain']
    datax_test = data['Xtest']
    datay_train = data['ytrain']
    datay_test = data['ytest']

    print(datax_train.shape, datax_test.shape, datay_train.shape, datay_test.shape)

    datax = np.append(datax_train, datax_test, axis=0)
    datay = np.append(datay_train, datay_test, axis=0)
    print(datay_train.max() - datay_train.min())
    print(datay_test.max() - datay_test.min())

    minRMSECV = np.array([])
    for _ in range(21):
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
