#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk("movie_success_rate.csv.xls"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[52]:


import warnings
warnings.simplefilter('ignore')

import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve


# In[3]:


imdb_data = pd.read_csv("movie_success_rate.csv.xls")


# In[4]:


pd.set_option("Display.max_columns", None)
imdb_data


# In[5]:


#Data Exploration
imdb_data.shape


# In[6]:


imdb_data.info()


# In[7]:


imdb_data.describe()


# In[8]:


#Data Processing
imdb_data.isnull().sum()


# In[10]:


imdb_data=imdb_data.rename(columns = {'Revenue (Millions)':'Revenue_Millions'})
imdb_data=imdb_data.rename(columns = {'Runtime (Minutes)':'Runtime_Minutes'})


# In[11]:


import warnings
warnings.filterwarnings("ignore")


# In[12]:


imdb_data = imdb_data.dropna()


# In[13]:


imdb_data.isnull().sum()


# In[14]:


imdb_data = imdb_data.drop("Genre", axis = 1)


# In[15]:


imdb_data


# In[21]:


imd = imdb_data


# In[22]:


x = imd[imd.columns[6:32]]
y = imd["Success"]


# In[23]:


x


# In[24]:


y


# In[25]:


#DIRECTOR ANALYSIS :
imdb_data.Director.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))
plt.title('TOP 10 DIRECTORS OF MOVIES')
plt.show()


# In[26]:


#ACTOR ANALYSIS
imdb_data.Actors.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))
plt.title('TOP 10 Actors OF MOVIES')
plt.show()


# In[27]:


#YEAR ANALYSIS :
sns.stripplot(x="Year", y="Rating", data=imdb_data, jitter=True);
plt.title(' RATING BASED ON YEAR')
plt.show


# In[28]:


sns.swarmplot(x="Year", y="Votes", data=imdb_data);
plt.title(' VOTES BASED ON YEAR')


# In[29]:


#VOTES BASED ON YEAR
sns.stripplot(x="Year", y="Revenue_Millions", data=imdb_data, jitter=True);
plt.title(' REVENUE BASED ON YEAR')


# In[30]:


sns.swarmplot(x="Year", y="Metascore", data=imdb_data);
plt.title(' METASCORE BASED ON YEAR')


# In[31]:


#RATING ANALYSIS :

imdb_data["Rating"].value_counts()

#top 10 rating movies 
Sortedrating= imdb_data.sort_values(['Rating'], ascending=False)

#medium rated movies
mediumratedmovies= imdb_data.query('(Rating > 3.0) & (Rating < 7.0)')

#high rated movies
highratedmovies= imdb_data.query('(Rating > 7.0) & (Rating < 10.0)')


# In[32]:


sns.jointplot(x="Rating", y="Metascore", data=mediumratedmovies);
plt.title('(MOVIES WITH MEDIUM RATING , METASCORE')


# In[33]:


sns.jointplot(x="Rating", y="Votes", data=mediumratedmovies);
plt.title('(MOVIES WITH MEDIUM RATING , VOTES')


# In[34]:


sns.jointplot(x="Rating", y="Revenue_Millions", data=mediumratedmovies);
plt.title('(MOVIES WITH MEDIUM RATING , REVENUE')


# In[35]:


sns.jointplot(x="Rating", y="Metascore", data=highratedmovies);
plt.title('(MOVIES WITH HIGH RATING , METASCORE')


# In[36]:


sns.jointplot(x="Rating", y="Votes", data=highratedmovies);
plt.title('(MOVIES WITH HIGH RATING ,VOTES')


# In[37]:


sns.jointplot(x="Rating", y="Revenue_Millions", data=highratedmovies);
plt.title('(MOVIES WITH HIGH RATING ,REVENUE')


# In[38]:


metascore=imdb_data.Metascore
sns.boxplot(metascore);
plt.show()


# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve


# In[40]:


x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)


# In[41]:


print(x.shape)
print(x_train.shape)
print(x_test.shape)


# In[42]:


scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[43]:


#MODELLING
rf = RandomForestClassifier(n_estimators=60)


# In[44]:


rf.fit(x_train, y_train)

y_pred_rf_test = rf.predict(x_test)

y_pred_rf_train = rf.predict(x_train)


# In[45]:


testacc_rf = accuracy_score(y_pred_rf_test, y_test)

trainacc_rf = accuracy_score(y_pred_rf_train, y_train)


# In[46]:


print("Training Accuracy of Random Forest: ", trainacc_rf)

print("Testing Accuracy of Random Forest: ", testacc_rf)


# In[47]:


sns.heatmap(confusion_matrix(y_test,y_pred_rf_test), fmt = 'd',annot=True, cmap='magma')


# In[48]:


print(classification_report(y_test,y_pred_rf_test))


# In[ ]:





# In[ ]:




