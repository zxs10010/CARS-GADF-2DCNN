import pandas as pd


def write_to_csv_1D(write_data,select_method,c):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    # df.to_csv('E:\\NIRATSpecNet-main\\soil\\'+select_method+'_1'+'\\'+'soil_1D_diff_'+select_method+'_'+str(c)+'.csv', mode='a', index=False, header=False)
    df.to_csv('soil_1D_3.csv', mode='a', index=False, header=False)
    return

def write_to_csv_2D_diff(write_data,select_method,c):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('E:\\NIRATSpecNet-main\\soil\\'+select_method+'_1'+'\\'+'soil_2D_diff_'+select_method+'_'+str(c)+'.csv', mode='a', index=False, header=False)
    # df.to_csv('soil_2D.csv', mode='a', index=False, header=False)
    return

def write_to_csv_2D_sum(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('soil_2D_sum.csv', mode='a', index=False, header=False)
    return

def write_RMSECVmin(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('RMSECVmin_5.csv', mode='a', index=False, header=False)
    return
