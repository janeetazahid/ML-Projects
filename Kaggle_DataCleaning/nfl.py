#imports
import pandas as pd 
import numpy as np


nfl_df=pd.read_csv('NFL Play by Play 2009-2017 (v4).csv')
np.random.seed(0)

#explore dataset 
print(nfl_df.head(5))

#count the number of missing data points we have
missing_value_count=nfl_df.isnull().sum() #returns the number of missing in each column
print(missing_value_count[0:10])

total_cells=np.product(nfl_df.shape) #total number of cells 
total_missing_cells=missing_value_count.sum()

precent_missing=(total_missing_cells/total_cells)*100 #precentage of missing cells 
print(precent_missing)

#filling in values automatically 
subset_nfl_df=nfl_df.loc[:,'EPA':'Season'].head() #coumns EPA to Season and the first 5 rows 
print(subset_nfl_df)

subset_filled_0=subset_nfl_df.fillna(0) #fill n/as with 0
print(subset_filled_0)

#fill all NAs with values that comes in next col and if no next col fill with 0
subset_filled_next=subset_nfl_df.fillna(method='bfill',axis=0).fillna(0)
print(subset_filled_next)
