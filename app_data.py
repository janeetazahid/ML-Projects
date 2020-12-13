


#imports 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import metrics 
import category_encoders as ce 

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
"""
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
#print(f"Test score: {score}")
"""
"""
#apply count encoder
count_enc = ce.CountEncoder(cols=categorical_features)
count_enc.fit(train[categorical_features])
train_encoded = train.join(count_enc.transform(train[categorical_features]).add_suffix('_count'))
valid_encoded = valid.join(count_enc.transform(valid[categorical_features]).add_suffix('_count'))

#train model
dtrain_CE=lgb.Dataset(train_encoded[feature_cols],label=train_encoded['is_attributed'])
dvalid_CE = lgb.Dataset(valid_encoded[feature_cols], label=valid_encoded['is_attributed'])
param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
#train model
bst_CE = lgb.train(param, dtrain_CE, num_round, valid_sets=[dvalid_CE], early_stopping_rounds=10)

#evalute model
ypred_CE = bst_CE.predict(valid_encoded[feature_cols])
score_CE= metrics.roc_auc_score(valid_encoded['is_attributed'], ypred_CE)
print(f"Test score: {score_CE}")
"""
"""
#apply target encoding
target_enc = ce.TargetEncoder(cols=categorical_features)
target_enc.fit(train[categorical_features],train['is_attributed'])
train_TE = train.join(target_enc.transform(train[categorical_features]).add_suffix('_target'))
valid_TE = valid.join(target_enc.transform(valid[categorical_features]).add_suffix('_target'))

#train model
dtrain_TE=lgb.Dataset(train_TE[feature_cols],label=train_TE['is_attributed'])
dvalid_TE = lgb.Dataset(valid_TE[feature_cols], label=valid_TE['is_attributed'])
param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
#train model
bst_TE = lgb.train(param, dtrain_TE, num_round, valid_sets=[dvalid_TE], early_stopping_rounds=10)

#evalute model
ypred_TE = bst_TE.predict(valid_TE[feature_cols])
score_TE= metrics.roc_auc_score(valid_TE['is_attributed'], ypred_TE)
print(f"Test score: {score_TE}")
"""

#remove IP column 
categorical_features_2=['app','device','os','channel']
cb_enc=ce.CatBoostEncoder(cols=categorical_features_2,random_state=7)
cb_enc.fit(train[categorical_features_2],train['is_attributed'])
train_CB = train.join(cb_enc.transform(train[categorical_features_2]).add_suffix('_cb'))
valid_CB = valid.join(cb_enc.transform(valid[categorical_features_2]).add_suffix('_cb'))

#train model
dtrain_CB=lgb.Dataset(train_CB[feature_cols],label=train_CB['is_attributed'])
dvalid_CB = lgb.Dataset(valid_CB[feature_cols], label=valid_CB['is_attributed'])
param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
#train model
bst_CB = lgb.train(param, dtrain_CB, num_round, valid_sets=[dvalid_CB], early_stopping_rounds=10)

#evalute model
ypred_CB= bst_CB.predict(valid_CB[feature_cols])
score_CB= metrics.roc_auc_score(valid_CB['is_attributed'], ypred_CB)
print(f"Test score: {score_CB}")