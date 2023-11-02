import configparser
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
import numpy as np

cf = configparser.ConfigParser()


def generate_file(appliance_name, meter_name):
    file_dir = "电吹风_电水壶.csv"
    df_align = pd.read_csv(file_dir, index_col=False)

    # aggregate_mean = 59.17851424911944
    aggregate_mean = 53.332337565347274
    # aggregate_std = 173.05423074036995
    aggregate_std = 159.84074671066946
    # appliance_min = 0.0
    appliance_min = 0.0
    # appliance_max = 1365.0
    appliance_max = 995.0

    df_align['aggregate'] = standardize_data(df_align['aggregate'], aggregate_mean, aggregate_std)
    df_align[appliance_name] = normalize_data(df_align[appliance_name], appliance_min, appliance_max)
    df_align = society_normalization(df_align)

    df_align.to_csv("./电吹风_电水壶_后.csv", index=False, header=False)


def society_normalization(df_align):
    # continues = ['centigrade', 'people']
    # cs = MinMaxScaler()
    # df_align[continues] = cs.fit_transform(df_align[continues])   # 这三句是正常的，原始数据有问题，后期可放开注释
    df_align = df_align.astype({'centigrade': 'float', 'people': 'int'})
    max_cen = df_align['centigrade'].max()
    max_peo = df_align['people'].max()
    df_align['centigrade'] = df_align['centigrade'] / max_cen
    df_align['people'] = df_align['people'] / max_peo
    lb = LabelBinarizer().fit(df_align['isworkday'])
    df_align['isworkday'] = lb.transform(df_align['isworkday'])
    return df_align


def standardize_data(data, mu=0.0, sigma=1.0):
    data -= mu
    data /= sigma
    return data


def normalize_data(data, min_value=0.0, max_value=1.0):
    data -= min_value
    diff = max_value - min_value
    if diff != 0:
        data /= diff
    return data


# meter = "180315000016"
meter = "489190910750"
# appliance = "澳柯玛冰柜"
appliance = "虎牌电热水壶"
generate_file(appliance, meter)
