import pandas as pd
from sqlalchemy import create_engine

child_meter_1 = ['489190910751', '489190910750', '180315000016', '160327039', '160327042']
child_meter_2 = ['489190910753']
child_meter_3 = ['489190910752', '489190910754']
meter_relation = {'489190910754': child_meter_1, "489190910752": child_meter_2, "70213": child_meter_3}

username = "root"
password = "root"
host = "localhost"
port = "3306"
db = "energy610"
engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(username, password, host, port, db))


# 生成总功率数据
def generate_mains_common(meter, engine):
    meter_table_name = 'meter_{}_source'.format(meter)
    sql_query = "select timestamp, KW from {}".format(meter_table_name)
    mains_df = data_read_database(engine, sql_query)
    mains_df.rename(columns={"KW": 'aggregate', "timestamp": "time"}, inplace=True)
    mains_df['time'] = mains_df['time'].astype('str')
    mains_df['aggregate'] = mains_df['aggregate'].astype('float64')
    mains_df['aggregate'] = mains_df['aggregate'] * 1000
    mains_df['time'] = pd.to_datetime(mains_df['time'], unit='ms')
    return mains_df


def generate_appliance_common(meter, engine):
    meter_table_name = 'meter_{}_source'.format(meter)
    sql_query = "select timestamp, KW from {}".format(meter_table_name)
    meter_df = data_read_database(engine, sql_query)
    meter_df.rename(columns={"KW": meter, "timestamp": "time"}, inplace=True)
    meter_df['time'] = meter_df['time'].astype('str')
    meter_df[meter] = meter_df[meter].astype('float64')
    meter_df[meter] = meter_df[meter] * 1000
    meter_df['time'] = pd.to_datetime(meter_df['time'], unit='ms')
    return meter_df


# 拼装总功率和单一电器的功率
def generate_mains_appliance(mains_df, meter_df, meter_name):
    mains_df.set_index('time', inplace=True)
    meter_df.set_index('time', inplace=True)
    df_align = mains_df.join(meter_df, how='outer')
    df_align = df_align.dropna()
    df_align.reset_index(inplace=True)
    df_align['time'] = df_align['time'].astype('str')
    df_align['aggregate'] = df_align['aggregate'].astype('float64')
    df_align[meter_name] = df_align[meter_name].astype('float64')
    del mains_df, meter_df
    return df_align


def data_read_database(engine, sql_query):
    df_read = pd.read_sql_query(sql_query, engine)
    return df_read


def start_compare():
    for main_meter in meter_relation.keys():
        child_list = meter_relation[main_meter]
        mains_df = generate_mains_common(main_meter, engine)
        mains_df['remain'] = mains_df['aggregate']
        for child_meter in child_list:
            appliance_df = generate_appliance_common(child_meter, engine)
            mains_df = generate_mains_appliance(mains_df, appliance_df, child_meter)
            mains_df['remain'] = mains_df['remain'] - mains_df[child_meter]
        print(mains_df)


start_compare()
