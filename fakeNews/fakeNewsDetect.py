#!/usr/bin/env python
# coding: utf-8

# In[39]:


#imports 
import numpy as np
import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[40]:


news_df=pd.read_csv('news.csv')
news_df.head()


# In[41]:


#target varible 
target_label=news_df.label


# In[42]:


target_label.head()


# In[43]:


#split data 
X_train,X_test,y_train,y_test=train_test_split(news_df['text'],target_label,test_size=0.2,random_state=42)


# In[44]:


print('X_test shape: {}, y_test shape: {}, X_train shape {}, y_train shape{}'.format(X_test.shape,y_test.shape,X_train.shape,y_train.shape))


# In[45]:


#create and train TFIDF vectorizer 
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(X_test)


# In[46]:


pac=PassiveAggressiveClassifier(max_iter=100)


# In[47]:


pac.fit(tfidf_train,y_train)


# In[48]:


y_pred=pac.predict(tfidf_test)


# In[49]:


score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[ ]:




