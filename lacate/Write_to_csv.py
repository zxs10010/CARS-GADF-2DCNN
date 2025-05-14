import pandas as pd


def write_to_csv_1D(write_data,select_method,c):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('F:\\NIRATSpecNet-main\\grain\\grain_lactate\\'+select_method+'\\'+'soil_1D_diff_'+select_method+'_'+str(c)+'.csv', mode='a', index=False, header=False)
    # df.to_csv('grain_lactate_1D_1.csv', mode='a', index=False, header=False)
    return

def write_to_csv_2D_diff(write_data,select_method,c):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    # df.to_csv('F:\\NIRATSpecNet-main\\grain\\grain_lactate\\'+select_method+'\\'+'soil_2D_diff_'+select_method+'_'+str(c)+'.csv', mode='a', index=False, header=False)
    # df.to_csv('grain_lactate_2D_diff_1.csv', mode='a', index=False, header=False)
    df.to_csv('result_predict_cars.csv', mode='a', index=False, header=False)
    return


def write_to_csv_2D_sum(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('grain_lactate_2D_sum.csv', mode='a', index=False, header=False)
    return
def write_RMSECVmin(write_data):
    df = pd.DataFrame(write_data)  # 列表数据转为数据框
    df.to_csv('RMSECVmin_6.csv', mode='a', index=False, header=False)
    return