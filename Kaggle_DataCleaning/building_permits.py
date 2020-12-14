#imports
import pandas as pd 
import numpy as np


permits_df=pd.read_csv('Building_Permits.csv')
np.random.seed(0)

#explore data 
print(permits_df.head())

#precentage of missing vlaues 
total_cols=np.product(permits_df.shape) #total cols 
total_missing=permits_df.isnull().sum().sum() #total missing vals
precent_missing=(total_missing/total_cols)*100

print(precent_missing)

#replace NAs with next values 
permits_df_replaced=permits_df.fillna(method='bfill',axis=0).fillna(0)
print(permits_df_replaced)