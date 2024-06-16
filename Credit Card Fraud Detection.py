#!/usr/bin/env python
# coding: utf-8

# import all library

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#  load the data set to use pandas dataframe

# In[4]:


data = pd.read_csv("creditcard.csv")


# In[6]:


data


# In[7]:


data.info()


# In[9]:


data.isnull().count()


# In[10]:


data.isnull().sum()


# Distribute of Legit transaction & fraudulent transaction

# In[12]:


data['Class'].value_counts()


# 0 -> Normal transaction
# 1 -> Fraudlent transaction

# In[14]:


normal = data[data.Class == 0]
fraud = data[data.Class == 1]


# In[15]:


normal


# In[16]:


fraud


# In[22]:


print(fraud.shape)
print(normal.shape)


# Statistical measures of the data

# In[24]:


normal.Amount.describe()


# In[29]:


fraud.Amount.describe()


# Compare the values for both transaction

# In[30]:


data.groupby('Class').mean()


# In[31]:


data.groupby('Amount').mean()


# Build a sample dataset containing similar distribution of normal transaction and fraudulent transaction

# In[33]:


normal_sample = normal.sample(n = 492)


# Concating two Dataframes

# In[38]:


new_dataset = pd.concat([normal_sample, fraud], axis = 0)


# In[39]:


new_dataset


# In[40]:


new_dataset.head(10)


# In[41]:


new_dataset.value_counts()


# In[42]:


new_dataset['Class'].value_counts()


# In[43]:


new_dataset.groupby('Class').mean()


# Splitting the data into features & targets

# In[44]:


x = new_dataset.drop(columns = 'Class', axis = 1)
y = new_dataset['Class']


# In[45]:


print(y)


# In[46]:


print(x)


# Split the data into Traning data & Testing data

# In[47]:


x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2) 


# In[48]:


print(x.shape, x_train.shape, x_test.shape)


# Model Training

# Logistic Regression

# In[49]:


model = LogisticRegression()


# traning the logistic regression model with training data

# In[58]:


model.fit(x_train, y_train)


# In[59]:


model.fit(x_test, y_test)


# Model Evaluation

# Accuracy Score

# In[60]:


x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)


# In[61]:


print(training_data_accuracy)


# In[65]:


x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)


# In[66]:


print(testing_data_accuracy)


# In[ ]:




