#imports 
# Using feature engineering to predict if a 
# Kickstarter project will succeed 

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import metrics 
import category_encoders as ce 


projects_df=pd.read_csv('ks-projects-201801.csv',parse_dates=['deadline', 'launched'])
#print(projects_df.head(5))
# print all of the columns 
#print(projects_df.columns)
# we want to predict the "state" column

## Data cleaning 
#dropping projects that are live 
projects_df=projects_df.query('state!="live"')
# in the state column, change enteries that say "sucessful"=1 and all others 0
projects_df=projects_df.assign(outcome=(projects_df['state']=='successful').astype(int))

#convert the laucnhed feature into categorical features 
#convert launced entry into hour, day, month and year
projects_df=projects_df.assign(hour=projects_df.launched.dt.hour,
                                day=projects_df.launched.dt.day,
                                month=projects_df.launched.dt.month,
                                year=projects_df.launched.dt.year)

#apply one hot encoding to categorical features 
categorical_features=['category','currency','country']
encoder=LabelEncoder()
encoded=projects_df[categorical_features].apply(encoder.fit_transform)

#create a new dataset with the encoded data and use it to train the model
data=projects_df[['goal','hour','day','month','year','outcome']].join(encoded)
#print(data.head())

#split the data into training, testing and validation sets 
valid_fraction=0.1
valid_size=int(len(data)*valid_fraction)
# 10% for validation, 10 for testing and 80% for training
train=data[:-2*valid_size]
valid=data[-2*valid_size:-valid_size]
test=data[-valid_size:]

#define feature columns
features_cols=train.columns.drop('outcome')
#create training Dataset object
dtrain=lgb.Dataset(train[features_cols],label=train['outcome'])
#create validation Dataset object
dvalid=lgb.Dataset(valid[features_cols],label=valid['outcome'])

#create parameters list, we will use a max of 64 leaves in one tree
#our objective is binary classification
param={'num_leaves':64,'objective':'binary'}
param['metric']='auc'
#1000 boosting iterations
num_rounds=1000
bst=lgb.train(param,dtrain,num_rounds,valid_sets=[dvalid],early_stopping_rounds=10,verbose_eval=False)

y_pred=bst.predict(test[features_cols])
score=metrics.roc_auc_score(test['outcome'],y_pred)

#print("Test AUC Score{}".format(score))

#try applying count encoding
count_encoder=ce.CountEncoder()
#train a new model with count encoded data
count_encoded=count_encoder.fit_transform(projects_df[categorical_features])
data_2=data.join(count_encoded.add_suffix("_count"))
train_2=data_2[:-2*valid_size]
test_2=data_2[-valid_size:]
valid_2=data_2[-2*valid_size:-valid_size]

dtrain_2=lgb.Dataset(train_2[features_cols],label=train_2['outcome'])
dvalid_2=lgb.Dataset(valid_2[features_cols],label=valid_2['outcome'])
bst_2=lgb.train(param,dtrain_2,num_boost_round=1000,valid_sets=[dvalid_2],early_stopping_rounds=10,verbose_eval=False)

valid_pred=bst_2.predict(valid_2[features_cols])
valid_score=metrics.roc_auc_score(valid_2['outcome'],valid_pred)

print("Validation AUC score: {}".format(valid_score))
