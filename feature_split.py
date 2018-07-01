import pandas as pd
import numpy as np

data_feature_name = pd.read_csv('data_0608/feature_0525_col.csv', delimiter='\t', names=[''])
data_feature = pd.read_csv('data_0619/feature_0626_10_21.csv', delimiter='\t',
                       names=[x.strip() for x in list(data_feature_name.reset_index()['level_0'])[:-5]])


data_feature[data_feature['dt']==20180521].to_csv('data_0619/feature_0521.csv')
data_feature[data_feature['dt']==20180520].to_csv('data_0619/feature_0520.csv')
data_feature[data_feature['dt']==20180519].to_csv('data_0619/feature_0519.csv')
data_feature[data_feature['dt']==20180518].to_csv('data_0619/feature_0518.csv')
data_feature[data_feature['dt']==20180517].to_csv('data_0619/feature_0517.csv')
data_feature[data_feature['dt']==20180516].to_csv('data_0619/feature_0516.csv')
data_feature[data_feature['dt']==20180515].to_csv('data_0619/feature_0515.csv')
data_feature[data_feature['dt']==20180514].to_csv('data_0619/feature_0514.csv')
data_feature[data_feature['dt']==20180513].to_csv('data_0619/feature_0513.csv')
data_feature[data_feature['dt']==20180512].to_csv('data_0619/feature_0512.csv')
data_feature[data_feature['dt']==20180511].to_csv('data_0619/feature_0511.csv')
