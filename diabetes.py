
# coding: utf-8

# In[92]:


import numpy as np 
import pandas as pd


# In[93]:


data = pd.read_csv('G:/ml/diabetes/diabetes.csv')


# In[94]:


len(data[data['Outcome']==1])


# In[95]:


print(100*np.mean(data['Outcome'][data['Age']>50]))
print(100*np.mean(data['Outcome'][data['Age']>50]))


# In[96]:


print(100*np.mean(data['Outcome'][data['Pregnancies']<5]))
print(100*np.mean(data['Outcome'][data['Pregnancies']>=5]))


# In[97]:


features = data.iloc[:,0:8]
target=data.iloc[:,8:9]


# In[98]:


features.head()


# In[99]:


features['Pregnancies']=features['Pregnancies'].fillna(np.mean(features['Pregnancies']))


# In[100]:


nulls = pd.DataFrame(features.isnull().sum().sort_values(ascending =False)[:])
nulls.columns = ['Null count']
nulls.index.name = 'Feature'
nulls


# In[101]:


from sklearn.model_selection import train_test_split
features_train,features_test,target_train,target_test =train_test_split(features,target,random_state = 42,test_size=0.11)


# In[102]:


from sklearn import linear_model
clf =linear_model.LogisticRegression(random_state=0,max_iter=100)
clf.fit(features_train,target_train)


# In[103]:


from sklearn.metrics import accuracy_score
print(accuracy_score(target_train,clf.predict(features_train)))
print(accuracy_score(target_test,clf.predict(features_test)))

