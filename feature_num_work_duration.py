import json
import jieba
import re
import numpy as np
import pandas as pd
from zhconv import convert
from datetime import datetime

def load_csv_data(data_path):
    '''
    读取csv文件
    '''
    df = pd.read_csv(data_path)
    return df

def parse_ym(s):
    year_s, mon_s = s.split('-')
    return datetime(int(year_s), int(mon_s), 1)

def get_job_time_fun(series, col):
    '''
    从jobTracks列提取工作的开始时间、结束时间和时间差(天数)
    '''
    time_pair_list = []
    try:
        jobtrack = series[col]
        jobtrack = jobtrack.replace("\'", "")
        jobtrack_list_dict = json.loads(jobtrack)

        for dic in jobtrack_list_dict:
            startDate = dic.get('startDate', None)
            endDate = dic.get('endDate', None)
        
            if not startDate or not endDate:
                continue

            start = parse_ym(startDate)
            end = parse_ym(endDate)

            # 这个down_limit该如何定呢？低频+生活实际
            down_limit = parse_ym('1990-01')
            up_limit = datetime.now()

            if end > up_limit or start < down_limit: 
                continue

            if end <= start:
                continue

            delta = (end - start).days
            # 时间差的上限，经过分位点统计得到
            if delta > 3650:
                 continue
            
            time_pair_list.append((startDate, endDate, delta))
        
        return time_pair_list
    except:
        return []

def work_duration_mean_fun(series, col):
    '''
    对各份工作时长取均值
    '''
    job_time_delta = series[col]
    time_delta = [days for _,_,days in job_time_delta]

    return np.mean(time_delta)

def get_work_duration_mean(data_path):
    '''
    给定数据路径，得到work_duration_mean列
    '''
    all_data = load_csv_data(data_path)
    all_data['job_time_delta'] = all_data.apply(get_job_time_fun, axis=1, args=('jobTracks', ))
    all_data['work_duration_mean'] = all_data.apply(work_duration_mean_fun, axis=1, args=('job_time_delta', ))
    return all_data


if __name__ == "__main__":
    print("running...")

    data_path = '../data/all_sample_20220821_spark.csv'
    all_data = get_work_duration_mean(data_path)

    print(all_data[['jobTracks', 'job_time_delta', 'work_duration_mean']])
    print(all_data[['jobTracks', 'job_time_delta', 'work_duration_mean']].info())

    all_data[['work_duration_mean']].to_csv('../data/work_duration_mean.csv')