import pandas as pd


def write_to_csv_1D(write_data,select_method,c):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    # df.to_csv('E:\\NIRATSpecNet-main\\grain_casein\\'+select_method+'_1'+'\\'+'pill1_1D_diff_'+select_method+'_'+str(c)+'.csv', mode='a', index=False, header=False)
    df.to_csv('pill_1D_3.csv', mode='a', index=False, header=False)
    return

def write_to_csv_2D_diff(write_data,select_method,c):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    # df.to_csv('E:\\NIRATSpecNet-main\\grain_casein\\'+select_method+'_1'+'\\'+'pill1_2D_diff_'+select_method+'_'+str(c)+'.csv', mode='a', index=False, header=False)
    # df.to_csv('pill_2D_3.csv', mode='a', index=False, header=False)
    df.to_csv('result_predict_cars.csv', mode='a', index=False, header=False)
    return

def write_to_csv_2D_sum(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('grain_casein_2D_sum.csv', mode='a', index=False, header=False)
    return

def write_RMSECVmin(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('RMSECVmin_5.csv', mode='a', index=False, header=False)
    return
