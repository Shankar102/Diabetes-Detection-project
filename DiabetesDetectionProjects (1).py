#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction using Machine Learning
# 
# Diabetes, is a group of metabolic disorders in which there are high blood sugar levels over a prolonged period. Symptoms of high blood sugar include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many complications. Acute complications can include diabetic ketoacidosis, hyperosmolar hyperglycemic state, or death. Serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, and damage to the eyes.
# 
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
# ## Objective
# We will try to build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?
# 
# ## **Details about the dataset:**
# 
# The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# 
# - **Pregnancies**: Number of times pregnant
# - **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# - **BloodPressure**: Diastolic blood pressure (mm Hg)
# - **SkinThickness**: Triceps skin fold thickness (mm)
# - **Insulin**: 2-Hour serum insulin (mu U/ml)
# - **BMI**: Body mass index (weight in kg/(height in m)^2)
# - **DiabetesPedigreeFunction**: Diabetes pedigree function
# - **Age**: Age (years)
# - **Outcome**: Class variable (0 or 1)
# 
# **Number of Observation Units: 768**
# 
# **Variable Number: 9**
# 
# **Result; The model created as a result of XGBoost hyperparameter optimization became the model with the lowest Cross Validation Score value. (0.90)**

# In[1]:


from PIL import Image
Img=Image.open('diabetes-blog.jpg')
Img


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Import Dataset

# In[3]:


df=pd.read_csv("Diabetes.csv")
df.head(5)


# # Explotary Data Analysis

# In[4]:


df.size


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T


# In[8]:


df.isnull().sum()


# In[9]:


df['Outcome'].value_counts()*100/len(df)


# In[10]:


df.Outcome.value_counts()


# In[11]:


df['BMI'].hist(edgecolor="black");


# In[12]:


df['Age'].hist(edgecolor="black");


# In[13]:


print("Maximum age:"+str(df['Age'].max()),"Minimum age:"+str(df['Age'].min()))


# In[14]:


fig,ax=plt.subplots(4,2,figsize=(16,16))
sns.distplot(df.Age,bins=20,ax=ax[0,0])
sns.distplot(df.Pregnancies,bins=20,ax=ax[0,1])
sns.distplot(df.Glucose,bins=20,ax=ax[1,0])
sns.distplot(df.BloodPressure,bins=20,ax=ax[1,1])
sns.distplot(df.SkinThickness,bins=20,ax=ax[2,1])
sns.distplot(df.Insulin,bins=20,ax=ax[2,0])
sns.distplot(df.BMI,bins=20,ax=ax[3,1])
sns.distplot(df.DiabetesPedigreeFunction,bins=20,ax=ax[3,0])
 


# In[15]:


df.groupby('Outcome').agg({"Pregnancies":"mean"})


# In[16]:


df.groupby('Outcome').agg({"Age":"mean"})


# In[17]:


df.groupby('Outcome').agg({"Age":"max"})


# In[18]:


df.groupby('Outcome').agg({"BMI":"max"})


# In[19]:


df.groupby('Outcome').agg({"BMI":"mean"})


# In[20]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Target')
ax[0].set_xlabel('Outcome')
sns.countplot(data=df,x='Outcome',ax=ax[1])
ax[1].set_xlabel("")
ax[1].set_title("outcome") 
plt.show()


# In[21]:


df.corr()


# In[22]:


f,ax=plt.subplots(figsize=[16,8])
sns.heatmap(df.corr(),ax=ax,annot=True,fmt=".2f",cmap="magma")


# # Missing Value Observation
# 

# In[23]:


df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]]=df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]].replace(0,np.NaN)


# In[24]:


df.isnull().sum()


# In[25]:


import missingno as msno
msno.bar(df)


# In[26]:


def median_var(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


# In[27]:


columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_var(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_var(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_var(i)[i][1]
    


# In[28]:


df.isnull().sum()


# In[29]:


for feature in df:
    
    Q1=df[feature].quantile(0.25)
    Q3=df[feature].quantile(0.75)
    
    IQR=Q3-Q1
    lower = Q1 - 1.15*IQR
    upper= Q3 + 1.15*IQR
    
    if df[(df[feature] > upper)].any(axis=None):
        print(feature,'Yes')
    else:
        print(feature,'No')
    
    
    
    


# In[30]:


fig,ax=plt.subplots(figsize=(16,8))
sns.boxplot(x=df['Insulin'],ax=ax);


# In[31]:


Q1=df.Insulin.quantile(0.25)
Q3=df.Insulin.quantile(0.75)

IQR=Q3-Q1
lower=Q1-1.15*IQR
upper=Q3+1.15*IQR

df.loc[df['Insulin'] > upper,"Insulin"] =upper


# In[32]:


sns.boxplot(x=df['Insulin'])


# # LocalOutlier Observation

# In[33]:


from sklearn.neighbors import LocalOutlierFactor
lof=LocalOutlierFactor(n_neighbors=10)
lof.fit_predict(df)


# In[34]:


df_score=lof.negative_outlier_factor_
np.sort(df_score)[0:30]


# In[35]:


thershold=np.sort(df_score)[7]
thershold


# In[36]:


outlier=df_score > thershold
df=df[outlier]


# # Feature Engineering

# In[37]:


NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]


# In[38]:


df.head(5)


# In[39]:


def set_insulin(var):
    if var["Insulin"] >=16 and var["Insulin"] <=166:
        return "Normal"
    else:
        return "Abnormal"


# In[40]:


df=df.assign(InsulinScore=df.apply(set_insulin,axis=1))

df.head(5)


# In[41]:


NewGlucose = pd.Series(['low','Normal','overwight','secret','high'],dtype='category')
df["NewGlucose"]=NewGlucose
df.loc[(df["Glucose"] > 70),"NewGlucose"]=NewGlucose[0]
df.loc[(df["Glucose"]>70) & (df["Glucose"] <=99),"NewGlucose"] =NewGlucose[1]
df.loc[(df["Glucose"]>99) & (df["Glucose"]<=126),"NewGlucose"]=NewGlucose[2]
df.loc[(df["Glucose"]>126),"NewGlucose"]=NewGlucose[3]


# In[42]:


df.head(5)


# In[43]:


df=pd.get_dummies(df,columns=["NewBMI","InsulinScore","NewGlucose"],drop_first=True)
df.head(5)


# In[44]:


cate_df=df[['NewBMI_Obesity 1','NewBMI_Obesity 2','NewBMI_Obesity 3','NewBMI_Overweight','NewBMI_Underweight','InsulinScore_Normal','NewGlucose_high','NewGlucose_low','NewGlucose_overwight','NewGlucose_secret']]
cate_df.head(5)


# # Split The Dataset Into Train & Test

# In[45]:


X=df.drop(["Outcome",'NewBMI_Obesity 1','NewBMI_Obesity 2','NewBMI_Obesity 3','NewBMI_Overweight','NewBMI_Underweight','InsulinScore_Normal','NewGlucose_high','NewGlucose_low','NewGlucose_overwight','NewGlucose_secret'],axis=1)
Y=df['Outcome']

cols=X.columns
index=X.index


# In[46]:


from sklearn.preprocessing import RobustScaler
transformer=RobustScaler().fit(X)
X=transformer.transform(X)
X = pd.DataFrame(X, columns = cols, index = index)
X.head(5
      )


# In[47]:


X=pd.concat([X,cate_df],axis=1)
X.head()


# In[48]:


Y.head(5)


# In[49]:


import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action = "ignore") 


# # Model Base

# In[50]:


models = []
models.append(('LR', LogisticRegression(random_state = 12345)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state = 12345)))
models.append(('RF', RandomForestClassifier(random_state = 12345)))
models.append(('SVM', SVC(gamma='auto', random_state = 12345)))
models.append(('XGB', GradientBoostingClassifier(random_state = 12345)))
models.append(("LightGBM", LGBMClassifier(random_state = 12345)))

# evaluate each model in turn
results = []
names = []


# In[51]:


for name, model in models:
    
        kfold = KFold(n_splits = 10, random_state = None)
        cv_results = cross_val_score(model, X, Y, cv = 10, scoring= "accuracy")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        

fig = plt.figure(figsize=(15,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
    


# # Model Tuning

# In[52]:


rf_par={"n_estimators":[100,200,500,1000],"max_depth":[3,5,8],"min_samples_split":[2,5,10,30],"max_features":[3,5,7]}



# In[53]:


rf_model=RandomForestClassifier(random_state=12345)


# In[54]:


gr_search=GridSearchCV(rf_model,rf_par, cv = 10,n_jobs = -1,verbose = 2).fit(X,Y)


# In[55]:


gr_search.best_params_


# In[56]:


rf_tun=RandomForestClassifier(**gr_search.best_params_)


# In[57]:


rf_tun= rf_tun.fit(X,Y)


# In[58]:


cross_val_score(rf_tun,X,Y,cv=10).mean()


# In[59]:


feature_imp=pd.Series(rf_tun.feature_importances_,index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp,y=feature_imp.index)


# In[60]:


lgbm=LGBMClassifier(random_state=12345)


# In[61]:


lbnm_par={"learning_rate":[0.01, 0.03, 0.05, 0.1, 0.5],"max_depth":(3,5,8),"n_estimators":[500,1000,1500]}


# In[62]:


gb_search=GridSearchCV(lgbm,lbnm_par,cv=10,n_jobs=-1,verbose=2).fit(X,Y)


# In[63]:


gb_search.best_params_


# In[64]:


gb_tun=LGBMClassifier(**gb_search.best_params_).fit(X,Y)


# In[65]:


cross_val_score(gb_tun,X,Y,cv=10).mean()


# In[66]:


features_imp=pd.Series(gb_tun.feature_importances_,index=X.columns).sort_values(ascending=False)

sns.barplot(x=features_imp,y=features_imp.index)


# In[67]:


xgb=GradientBoostingClassifier(random_state=(12345))


# In[68]:


xgb_par = {
    "learning_rate": [0.01, 0.1, 0.2, 1],
    "min_samples_split": np.linspace(0.1, 0.5, 10),
    "max_depth":[3,5,8],
    "subsample":[0.5, 0.9, 1.0],
    "n_estimators": [100,1000]}


# In[69]:


xgb_search=GridSearchCV(xgb,xgb_par,cv=10,n_jobs=-1,verbose=2).fit(X,Y)


# In[70]:


xgb_search.best_params_


# In[71]:


xgb_tun=GradientBoostingClassifier(**xgb_search.best_params_).fit(X,Y)


# In[72]:


cross_val_score(xgb_tun,X,Y,cv=10).mean()


# In[73]:


feature_imp=pd.Series(xgb_tun.feature_importances_,index=X.columns).sort_values(ascending =False)

sns.barplot(x=feature_imp,y=feature_imp.index)


# # Comeparison The Model

# In[74]:


models=[]

models.append(("RF",RandomForestClassifier(random_state=12345, max_depth= 8,
 max_features= 7,
 min_samples_split= 5,
 n_estimators= 500)))

models.append(("LGM",LGBMClassifier(random_state=12345, learning_rate= 0.01, max_depth= 8, n_estimators= 500)))

models.append(("XBG",GradientBoostingClassifier(random_state=12345,learning_rate= 0.2,
 max_depth= 3,
 min_samples_split= 0.14444444444444446,
 n_estimators= 100,
 subsample= 1.0)))

results=[]
names=[]


# In[75]:


for name, model in models:
    
        kfold = KFold(n_splits = 10, random_state = None)
        cv_results = cross_val_score(model, X, Y, cv = 10, scoring= "accuracy")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ## **Reporting:**

# 
# 
# Diabetes Dataset Insert
# 
# Explotary Data Anaylis: Anaysis the reletion one variable amoungs to other variables used the correlation and visualisation data set missing value 0 replaced with NaN.
# 
# Data Preprocessing: Identify some noisy data or missing value in dataset filled with median value so the outlier remove, our model looks prety good.from Data extract some feature helps of Diabetes parameter. 
# 
# Model Building: Logistic Regression, KNN, SVM, CART, Random Forests, XGBoost, LightGBM like using machine learning models Cross Validation Score were calculated. Later Random Forests, XGBoost, LightGBM hyperparameter optimizations optimized to increase Cross Validation value.
# 
# Accurecy: After implementing the all model the final output was Xgbooter has lowest Validation Score 

# In[ ]:




