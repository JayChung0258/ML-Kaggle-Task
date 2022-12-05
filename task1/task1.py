#!/usr/bin/env python
# coding: utf-8

# In[109]:


from xgboost import XGBClassifier

import pandas as pd
from numpy import reshape
import sklearn
from sklearn import metrics
import time


# In[110]:


data = pd.read_csv('train_task1.csv')


# In[111]:


train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)


# In[112]:


X_train = train_data.iloc[:, :5]
y_train = train_data.iloc[:, 5]

X_test = test_data.iloc[:, :5]
y_test = test_data.iloc[:, 5]

pred_data = pd.read_csv('test_nov28_task1_only_features.csv')


# In[113]:


from xgboost import XGBClassifier


# In[114]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)


# In[115]:


# build XGBClassifier model
xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3)
# train data with train_data
xgboostModel.fit(X_train, y_train)
# predict with the test_data
predicted = xgboostModel.predict(X_train)


# In[116]:


# 預測成功的比例
print('訓練集: ',xgboostModel.score(X_train,y_train))
print('測試集: ',xgboostModel.score(X_test,y_test))


# In[117]:


# predict the real given data
answer = xgboostModel.predict(pred_data)
answer = le.inverse_transform(answer)


# In[118]:


da = {"Id": [x for x in range(1, len(answer)+1)], "Category": answer}


# In[119]:


df = pd.DataFrame (da)


# In[120]:


df.to_csv(index=False)


# In[108]:


compression_opts = dict(method='zip',
                        archive_name='out.csv')  
df.to_csv('out.zip', index=False,
          compression=compression_opts)  


# In[ ]:





# In[ ]:




