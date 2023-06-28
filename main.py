#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')


# # IMPORTING MODELS

# In[71]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import r2_score
# for xgboost install the dependency
# pip install xgboost
from xgboost import XGBClassifier
from sklearn import metrics


# # IMPORTING DATASET FROM KAGGLE OF TESLA(2010-2020)

# In[5]:


df = pd.read_csv("TSLA.csv")
df.head()


# In[7]:


df.shape
# from this we got to know there are 2416 rows and 7 different columns


# In[11]:


df.info()
# description of the dataset


# # DATA ANALYSIS

# In[18]:


plt.figure()
plt.plot(df['Close'])
plt.title('TESLA CLOSE PRICE', fontsize=20)
plt.ylabel('price in dollars')
plt.show()


# In[19]:


# the data in the ‘Close’ column and that available in the ‘Adj Close’ column is the same
df[df['Close']== df['Adj Close']].shape


# In[20]:


# so we can delete Adj Close
df = df.drop(['Adj Close'], axis=1)


# In[22]:


# check for the null values if any are present in the data frame.
df.isnull().sum()
# This implies that there are no null values in the data set provided.


# In[29]:


df.keys()
features=['Open', 'High', 'Low', 'Close', 'Volume']


# In[33]:


plt.subplots()
 
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])
plt.show()


# In[34]:


plt.subplots()
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()


# # FEATURE ENGINEERING

# In[55]:


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
# Create new columns
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df


# In[56]:


df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()


# In[57]:


data_grouped = df.groupby('year').mean()
plt.subplots()
 
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()


# In[58]:


df.groupby('is_quarter_end').mean()


# In[59]:


df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# In[60]:


plt.pie(df['target'].value_counts().values,
        labels=[0, 1], autopct='%1.1f%%')
plt.show()


# In[62]:


plt.figure()
 
# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()


# # DATA SPLITTING AND NORMALIZATION

# In[64]:


features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']
 
scaler = StandardScaler()
features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=13)
print(X_train.shape, X_valid.shape)


# # MODEL DEVELOPMENT AND EVALUATION

# In[80]:


# using linear regression
model1 = LogisticRegression()
model1.fit(X_train, Y_train)
y_pred=model1.predict(X_valid)
print('Training Accuracy : ', metrics.r2_score(Y_valid,y_pred))


# In[79]:


metrics.plot_confusion_matrix(model1, X_valid, Y_valid)
plt.show()


# In[ ]:




