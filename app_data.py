


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


#pply label encoding 
categorical_features=['ip', 'app', 'device', 'os', 'channel']
encoder=LabelEncoder()
for feature in categorical_features:
    encoded=encoder.fit_transform(clicks[feature])
    clicks[feature+'_labels']=encoded

feature_cols = ['day', 'hour', 'minute', 'second', 
                'ip_labels', 'app_labels', 'device_labels',
                'os_labels', 'channel_labels']

valid_frac=0.1
clicks_srt=clicks.sort_values('click_time') #sort by click time
valid_rows=int(len(clicks_srt)*valid_frac)
train=clicks_srt[:-valid_rows*2]
valid=clicks_srt[-valid_rows*2:-valid_rows]
test=clicks_srt[-valid_rows:]

#create test, train, valid dataset
dtrain=lgb.Dataset(train[feature_cols],label=train['is_attributed'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])
param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
#train model
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)


#evalute model
ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['is_attributed'], ypred)
print(f"Test score: {score}")