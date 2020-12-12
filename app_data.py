


#imports 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import metrics 

click_data=pd.read_csv("train_sample.csv",parse_dates=["click_time"])

#convert click time column into day, hour, minute and second 
#copy dataset
clicks=click_data.copy()
clicks['day']= clicks['click_time'].dt.day.astype('uint8')
clicks['hour']=clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')
