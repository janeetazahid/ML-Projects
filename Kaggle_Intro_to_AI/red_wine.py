#Imports 
# The objective of this script is to predict the quaility 
# of the red wine given the physiochemical measurments 
import matplotlib.pyplot as plt 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers,callbacks 
from tensorflow.keras.callbacks import EarlyStopping
from IPython.display import display
from sklearn.metrics import accuracy_score

#load csv file 
red_wine_df=pd.read_csv('red-wine.csv')
#explore the data 



#plot untrained data, with random initilization
model_not_trained=keras.Sequential([
    layers.Dense(units=1,input_shape=[1])
])
x=tf.linspace(-1.0,1.0,100)
y=model_not_trained(x)


#split into training, validation and testing
df_train=red_wine_df.sample(frac=0.7,random_state=0) #70% of dataset for training
df_valid=red_wine_df.drop(df_train.index) 
df_test=df_valid.sample(frac=0.5,random_state=0) 
df_valid=df_valid.drop(df_test.index) #15% to valud and 15% to test


print(red_wine_df.shape)
print("Training size= {}, validation size ={}, testing size ={}".format(df_train.shape,df_valid.shape,df_test.shape))



X_train=df_train.drop('quality',axis=1) #drop target value 
X_valid=df_valid.drop('quality',axis=1)#drop target value 
X_test=df_test.drop('quality',axis=1)#drop target value 
y_train=df_train['quality'] #target value
y_valid=df_valid['quality'] #target value
y_test=df_test['quality'] #target value


#min delta= min amount of change to count as improvement, patience= how many epochs to wait before stopping
early_stopping= callbacks.EarlyStopping(min_delta=0.001,patience=20,restore_best_weights=True)
model=keras.Sequential([
    layers.Dense(512,activation='relu',input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024,activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024,activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1)
])
#compile model
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mse']
)
history=model.fit(
    X_train,y_train,
    validation_data=(X_valid,y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping],
    verbose=0
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()



