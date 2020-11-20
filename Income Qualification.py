#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('E:\Datasets\Machine-Learning--Projects-master\Projects\Projects for Submission\Project 2 - Income Qualification')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ### Problem Statement Scenario:
# Many social programs have a hard time making sure the right people are given enough aid. It’s tricky when a program focuses on the poorest segment of the population. This segment of population can’t provide the necessary income and expense records to prove that they qualify.
# 
# In Latin America, a popular method called Proxy Means Test (PMT) uses an algorithm to verify income qualification. With PMT, agencies use a model that considers a family’s observable household attributes like the material of their walls and ceiling or the assets found in their homes to classify them and predict their level of need. While this is an improvement, accuracy remains a problem as the region’s population grows and poverty declines.
# 
# The Inter-American Development Bank (IDB) believes that new methods beyond traditional econometrics, based on a dataset of Costa Rican household characteristics, might help improve PMT’s performance.
# 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# #### Let us explore our dataset before moving further

# In[5]:


print('Shape of train dataset is {}'.format(train.shape))
print('Shape of test dataset is {}'.format(test.shape))


# ### Let us identify our target variable

# In[6]:


for i in train.columns:
    if i not in test.columns:
        print("Our Target variable is {}".format(i))


# ### Lets Understand the type of data.

# In[7]:


print(train.dtypes.value_counts())


# In[8]:


print(train.info())


# **We have mixed data types. Specified as below:**
# 
# * float64 : 8 variables
# * int64 : 130 vriables
# * object :5 variables

# In[9]:


#lets explore each different types of datasets
for i in train.columns:
    a=train[i].dtype
    if a == 'object':
        print(i)


# Below is Data dictionary for above object variables
# * ID = Unique ID
# * idhogar, Household level identifier
# * dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
# * edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# * edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# 
# 

# In[10]:


# lets drop Id variable.

train.drop(['Id','idhogar'],axis=1,inplace=True)


# In[11]:


train['dependency'].value_counts()


# **Lets Convert object variables into numerical data.**

# In[12]:


def map(i):
    
    if i=='yes':
        return(float(1))
    elif i=='no':
        return(float(0))
    else:
        return(float(i))


# In[13]:


train['dependency']=train['dependency'].apply(map)


# In[14]:


for i in train.columns:
    a=train[i].dtype
    if a == 'object':
        print(i)


# In[15]:


train.info()


# In[16]:


train['edjefe']=train['edjefe'].apply(map)
train['edjefa']=train['edjefa'].apply(map)


# In[17]:


train.info()


# Now all data is in numerical form

# **Lets identify variable with 0 varinace**

# In[18]:


var_df=pd.DataFrame(np.var(train,0),columns=['variance'])
var_df.sort_values(by='variance').head(15)
print('Below are columns with variance 0.')
col=list((var_df[var_df['variance']==0]).index)
print(col)


#  elimbasu5 : 1 if rubbish disposal mainly by throwing in river, creek or sea.  
#  
# Interpretation :***From above it is shown that all values of elimbasu5 is same so there is no variablity in dataset therefor we will drop this variable***

# ### Check if there are any biases in your dataset.

# In[19]:


contingency_tab=pd.crosstab(train['r4t3'],train['hogar_total'])
Observed_Values=contingency_tab.values
import scipy.stats
b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]
no_of_rows=len(contingency_tab.iloc[0:2,0])
no_of_columns=len(contingency_tab.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
alpha=0.05
critical_value=chi2.ppf(q=1-alpha,df=df)
print('critical_value:',critical_value)
p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',df)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# Therefore,variables ('r4t3','hogar_total') have relationship between them. For good result we can use any one of them.

# In[20]:


contingency_tab=pd.crosstab(train['tipovivi3'],train['v2a1'])
Observed_Values=contingency_tab.values
import scipy.stats
b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]
no_of_rows=len(contingency_tab.iloc[0:2,0])
no_of_columns=len(contingency_tab.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
alpha=0.05
critical_value=chi2.ppf(q=1-alpha,df=df)
print('critical_value:',critical_value)
p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',df)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# Therefore,variables ('tipovivi3','v2a1') have relationship between them. For good result we can use any one of them.

# In[21]:


contingency_tab=pd.crosstab(train['v18q'],train['v18q1'])
Observed_Values=contingency_tab.values
import scipy.stats
b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]
no_of_rows=len(contingency_tab.iloc[0:2,0])
no_of_columns=len(contingency_tab.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
alpha=0.05
critical_value=chi2.ppf(q=1-alpha,df=df)
print('critical_value:',critical_value)
p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',df)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# Therefore,variables ('v18q','v18q1') have relationship between them. For good result we can use any one of them.

# Conclusion : ***Therefore, there is bias in our dataset.***

# In[22]:


train.drop('r4t3',axis=1,inplace=True)


# ### Check if there is a house without a family head.

# "parentesco1" =1 if household head

# In[23]:


train.parentesco1.value_counts()


# In[24]:


pd.crosstab(train['edjefa'],train['edjefe'])


# Interpretation : ***Above cross tab shows 0 male head and 0 female head which implies that there are 435 families with no family head.***

# ### Count how many null values are existing in columns.

# In[25]:


train.isna().sum().value_counts()


# Lets Identify number of null values in Target variable

# In[26]:


train['Target'].isna().sum()


# Interpretation : **There are no null values in Target variable. Now lets proceed further and identify and fillna of other variable.**

# In[27]:


float_col=[]
for i in train.columns:
    a=train[i].dtype
    if a == 'float64':
        float_col.append(i)
print(float_col)


# In[28]:


train[float_col].isna().sum()


# In[29]:


train['v18q1'].value_counts()


# In[30]:


pd.crosstab(train['tipovivi1'],train['v2a1'])


# In[31]:


pd.crosstab(train['v18q1'],train['v18q'])


# Interpretation and action : ***'v2a1', 'v18q1', 'rez_esc' have more than 50% null values, because for v18q1, there are families with their own house so they won't pay rent in that case it should be 0 and similar is for v18q1 there can be families with 0 tablets.***
# 
# 
# ***Istead we can drop a column tipovivi3,v18q***
# 
# * tipovivi3, =1 rented
# * v18q, owns a tablet
# 
# **as v2a1 alone can show both**
# **as v18q1 alone can show that if respondent owns a tablet or not

# In[32]:


train['v2a1'].fillna(0,inplace=True)
train['v18q1'].fillna(0,inplace=True)


# In[33]:


train.drop(['tipovivi3', 'v18q','rez_esc','elimbasu5'],axis=1,inplace=True)


# In[34]:


train['meaneduc'].fillna(np.mean(train['meaneduc']),inplace=True)
train['SQBmeaned'].fillna(np.mean(train['SQBmeaned']),inplace=True)
print(train.isna().sum().value_counts())


# In[35]:


int_col=[]
for i in train.columns:
    a=train[i].dtype
    if a == 'int64':
        int_col.append(i)
print(int_col)


# In[36]:


train[int_col].isna().sum().value_counts()


# Interpretation : ***Now there is no null value in our datset.***

# In[37]:


train.Target.value_counts()


# ### Set the poverty level of the members and the head of the house same in a family.
# 
# Now for people below poverty level can be people paying less rent and don't own a house. and it also depends on whether a house is in urban area or rural area. 

# In[38]:


Poverty_level=train[train['v2a1'] !=0]


# In[39]:


Poverty_level.shape


# In[40]:


poverty_level=Poverty_level.groupby('area1')['v2a1'].apply(np.median)


# In[41]:


poverty_level


# * For rural area level if people paying rent less than 8000 is under poverty level. 
# * For Urban area level if people paying rent less than 140000 is under poverty level.

# In[42]:


def povert(x):
    if x<8000:
        return('Below poverty level')
    
    elif x>140000:
        return('Above poverty level')
    elif x<140000:
        return('Below poverty level: Ur-ban ; Above poverty level : Rural ')
   


# In[43]:


c=Poverty_level['v2a1'].apply(povert)


# In[44]:


c.shape


# In[45]:


pd.crosstab(c,Poverty_level['area1'])


# Interpretation : 
# * ***There are total 1242 people above poverty level independent of area whether rural or Urban***
# * ***Remaining 1111 people level depends on their area***
# 
# 
# **Rural :**
# 
# Above poverty level= 445
# 
# 
# **Urban :** 
# 
# Above poverty level =1103
# 
# Below poverty level=1081

# In[46]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[47]:


X_data=train.drop('Target',axis=1)
Y_data=train.Target


# In[48]:


X_data_col=X_data.columns


# ### Applying Standard Scalling to dataset

# In[49]:


from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_data_1=SS.fit_transform(X_data)
X_data_1=pd.DataFrame(X_data_1,columns=X_data_col)


# ### Now we will proceed to model fitting

# In[50]:


X_train,X_test,Y_train,Y_test=train_test_split(X_data_1,Y_data,test_size=0.25,stratify=Y_data,random_state=0)


# Lets identify best parameters for our model using GridSearchCv

# In[51]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier(random_state=0)
parameters={'n_estimators':[10,50,100,300],'max_depth':[3,5,10,15]}
grid=zip([rfc],[parameters])

best_=None

for i, j in grid:
    a=GridSearchCV(i,param_grid=j,cv=3,n_jobs=1)
    a.fit(X_train,Y_train)
    if best_ is None:
        best_=a
    elif a.best_score_>best_.best_score_:
        best_=a
        
        
print ("Best CV Score",best_.best_score_)
print ("Model Parameters",best_.best_params_)
print("Best Estimator",best_.best_estimator_)


# In[52]:


RFC=best_.best_estimator_
Model=RFC.fit(X_train,Y_train)
pred=Model.predict(X_test)


# In[53]:


print('Model Score of train data : {}'.format(Model.score(X_train,Y_train)))
print('Model Score of test data : {}'.format(Model.score(X_test,Y_test)))


# In[54]:


Important_features=pd.DataFrame(Model.feature_importances_,X_data_col,columns=['feature_importance'])


# In[55]:


Top50Features=Important_features.sort_values(by='feature_importance',ascending=False).head(50).index


# In[56]:


Top50Features


# In[57]:


for i in Top50Features:
    if i not in X_data_col:
        print(i)


# In[58]:


X_data_Top50=X_data[Top50Features]


# In[59]:


X_train,X_test,Y_train,Y_test=train_test_split(X_data_Top50,Y_data,test_size=0.25,stratify=Y_data,random_state=0)


# In[60]:


Model_1=RFC.fit(X_train,Y_train)
pred=Model_1.predict(X_test)


# In[61]:


from sklearn.metrics import confusion_matrix,f1_score,accuracy_score


# In[62]:


confusion_matrix(Y_test,pred)


# In[63]:


f1_score(Y_test,pred,average='weighted')


# In[64]:


accuracy_score(Y_test,pred)


# ### Lets apply cleaning on test data and then find prediction for that.

# In[65]:


# lets drop Id variable.
test.drop('r4t3',axis=1,inplace=True)
test.drop(['Id','idhogar'],axis=1,inplace=True)
test['dependency']=test['dependency'].apply(map)
test['edjefe']=test['edjefe'].apply(map)
test['edjefa']=test['edjefa'].apply(map)


# In[66]:


test['v2a1'].fillna(0,inplace=True)
test['v18q1'].fillna(0,inplace=True)


# In[67]:


test.drop(['tipovivi3', 'v18q','rez_esc','elimbasu5'],axis=1,inplace=True)


# In[68]:


train['meaneduc'].fillna(np.mean(train['meaneduc']),inplace=True)
train['SQBmeaned'].fillna(np.mean(train['SQBmeaned']),inplace=True)


# In[69]:


test_data=test[Top50Features]


# In[70]:


test_data.isna().sum().value_counts()


# In[71]:


test_data.SQBmeaned.fillna(np.mean(test_data['SQBmeaned']),inplace=True)


# In[72]:


test_data.meaneduc.fillna(np.mean(test_data['meaneduc']),inplace=True)


# In[73]:



Test_data_1=SS.fit_transform(test_data)
X_data_1=pd.DataFrame(Test_data_1)


# In[74]:


test_prediction=Model_1.predict(test_data)


# In[75]:


test_prediction


# Interpretation : ***Above is our prediction for test data.***

# ## Conclusion : 
# 
# ***Using RandomForest Classifier we can predict test_data with accuracy of 90%.***
