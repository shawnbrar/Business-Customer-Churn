#!/usr/bin/env python
# coding: utf-8

# First we import the important libraries - numpy and pandas.

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px


# Next, we load the data and have a look at the first 5 rows of the data. From the data we can see that the `Churn` variable which is our dependent variable.

# In[4]:


data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()


# Currently, we have no information about the data. Hence, we will now look at the type of columns the data has. This will tell us if our data has some categorical variable or not. Also, a lot of time some variables are also mistaken for different data types. This step will help us in catching that. We will also look at how many  categories each categorial data type has.

# In[5]:


print(data.dtypes)
print(data.select_dtypes("object").nunique())


# From the output we can see that there are categorical variables which have the data type `object`. From the second table, we can see the number of categories each categorical column has. Here we see that `TotalCharge` has `6531` types of categories, which does not makes sense. This is because it should be a numerical variable, i.e., have data types like `int` or `float`. This means that this variable has been parsed as a string variable. Hence, we will correct this by changing it to a numeric data type. Also, certain times when one changes the data type from string to a numeric data type.

# In[ ]:


## Checking how many rows are null or not
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors = "coerce").fillna(0)


# In[6]:



## Checking how many rows are null or not
print((data.isnull().sum(axis = 0) == 0).all()) ## No columns have NA values
# Checking the number of unique customers
print(data.customerID.unique().size == data.shape[0]) ## All rows are unique customers


# In[7]:


fig = px.scatter(x=data['tenure'], y=data['TotalCharges'], 
                 color = data['Churn'], template = 'presentation', 
                 opacity = 0.5, facet_col = data['Contract'], 
                 title = 'Customer Churn by Tenure, Charges, and Contract Type',
                 labels = {'x' : 'Customer Tenure', 'y' : 'Total Charges $'})
fig.show()


# # Data Cleaning and Transformation

# In[8]:


data["MultipleLines"] = data["MultipleLines"].replace(['No phone service'], "No")
colz = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
data[colz] = data[colz].replace(['No internet service'], 'No')

data.gender = np.where(data.gender == "Male", 1, 0)
# Make variables with two classes into dummy variables
data.iloc[:, [3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 16, 20]] = np.where(data.iloc[:, [3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 16, 20]] == "Yes", 1, 0)

## Creating dummies for variables with more than two classes
data = pd.get_dummies(data, columns = ['InternetService', 'Contract', 'PaymentMethod'])
data.drop(data.columns[[0, 20, 21, 24]], axis=1, inplace=True) ## drop one base class and customerID


# Hence, a company should try to promote their one year/two year contracts as people with those contract are less likely churn.
# 
# # Data Preprocessing

# In[9]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

y = data["Churn"]
data.drop(["Churn"], axis = 1, inplace=True)

colz = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = preprocessing.StandardScaler().fit(data[colz])
data[colz] = scaler.transform(data[colz])

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=0)


# # Classifiers - Model Comparisons

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

models = [
    ('LR', LogisticRegression(random_state=0)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=0)),
    ('NB', GaussianNB()),
    ('SVM', SVC(random_state=0)),
    ("RF", RandomForestClassifier(n_estimators=150, min_samples_leaf=50, oob_score=True, n_jobs=-1, random_state=0)),
    ("RC", RidgeClassifier(random_state=0)),
    ("ETC", ExtraTreeClassifier(random_state=0)),
    ("QDA", QuadraticDiscriminantAnalysis())
]


# In[11]:


names = []
test_results = []
for name, model in models:
    model.fit(X_train, y_train)
    rez = model.score(X_test, y_test)
    test_results.append(rez)
    names.append(name)
    msg = "%s: %f" % (name, rez)
    print(msg)


# In[12]:


import matplotlib.pyplot as plt

plt.scatter(names, test_results)


# In[13]:


score_res = pd.DataFrame(list(zip(names, test_results, range(0, len(models)))), columns = ["Model", "Score", "Index"])
score_res.sort_values(by = "Score", ascending = False, inplace=True)


# # Hyperparameter Tuning
# ### Grid Search CV

# In[14]:


from sklearn.model_selection import GridSearchCV
searchspace = {"alpha": np.linspace(10, 100, 10)}
#searchspace = {'alpha' : [0.1, 0.2, 0.4, 0.8]}
GS = GridSearchCV(models[score_res.iloc[0, 2]][1], ## model with the highest accuracy
                 param_grid=searchspace,
                  scoring=["accuracy"],
                  refit="accuracy",
                  cv=5,
                  verbose=4
                 )
GS.fit(X_train, y_train)


# In[15]:


print(GS.best_params_)
print(GS.best_score_)


# In[16]:


pd.DataFrame(GS.cv_results_)


# In[17]:


selected_model = RidgeClassifier(alpha = 100, random_state=0)
selected_model.get_params()
selected_model.fit(X_train, y_train)
predicted_y = selected_model.predict(X_test)
print(selected_model.score(X_test, y_test))
print(confusion_matrix(y_test, predicted_y))


# ### Feature Importance

# In[18]:


from sklearn.inspection import permutation_importance
r = permutation_importance(selected_model, X_test, y_test, n_repeats=30, random_state=0, scoring="accuracy")


# In[19]:


for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{data.columns[i]:<8}"
        f"{r.importances_mean[i]:.3f}"
        f" +/- {r.importances_std[i]:.3f}")

importance = pd.DataFrame(list(zip(data.columns, r.importances_mean, r.importances_std)), columns = ["Feature", "ImportanceMean", "ImportanceStd"])
importance.sort_values(by = "ImportanceMean", ascending=False, inplace=True)
importance


# In[20]:


importance = pd.Series(r.importances_mean, index = data.columns)


# In[21]:


importance.nlargest(10).sort_values(ascending=False).plot(kind = "barh")


# # Adding a New Metric

# In[22]:


from sklearn.metrics import make_scorer
# Profit Metric
def profit(y, y_pred):
    tp = np.where((y_pred==1) & (y==1), (5000-1000), 0)
    fp = np.where((y_pred==1) & (y==0), -1000, 0)
    return np.sum([tp,fp])

profit_scorer = make_scorer(profit)


# In[23]:


prf_names = []
prf_test_results = []
for name, model in models:
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    rez = profit(y_test, y_predicted)
    prf_test_results.append(rez)
    prf_names.append(name)
    msg = "%s: %f" % (name, rez)
    print(msg)


# In[24]:


plt.scatter(prf_names, prf_test_results)


# In[25]:


prf_score_res = pd.DataFrame(list(zip(prf_names, prf_test_results, range(0, len(models)))), columns = ["Model", "Profit", "Index"])
prf_score_res.sort_values(by = "Profit", ascending = False, inplace=True)
prf_score_res


# In[26]:


confusion_matrix(y_test, models[prf_score_res.iloc[0, 2]][1].predict(X_test))

