#Imports 
# The objective of this script is to predict the quaility 
# of the red wine given the physiochemical measurments 
import matplotlib.pyplot as plt 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 


#load csv file 
red_wine_df=pd.read_csv('red-wine.csv')
#explore the data 
print(red_wine_df.head())

#set up model
model=keras.Sequential([
    layers.Dense(units=1,input_shape=[11])
])
#check the untrained weights 
w,b=model.weights

print("The untrained weights are: {} and the Bias is {}".format(w,b))
#plot untrained data, with random initilization
model_not_trained=keras.Sequential([
    layers.Dense(units=1,input_shape=[1])
])
x=tf.linspace(-1.0,1.0,100)
y=model_not_trained(x)

plt.figure(dpi=100)
plt.plot(x,y,'k')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w,b=model_not_trained.get_weights()
plt.title("Weight {:0.2f}\nBias:{:0.2f}".format(w[0][0],b[0]))
plt.show()
