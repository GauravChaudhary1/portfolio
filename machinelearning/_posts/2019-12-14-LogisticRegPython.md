---
title: "Logistic Regression Implementation using Python"
categories: [machinelearning]
tags: [logisticregression,python]
---
#### Logistic Regression <br>
Basic Assumptions: <br>
    1) target variable y is binary and follows Bernoulli distribution <br>
    2) independent variable should have little or no collinearity <br>
    3) linearly related to the log odds <br>
    4) should have sufficient amount of data <br>
    5) Model parameters are estimated using gradient descent or Max Conditional Likelihood estimation

#### Problem Statement
The objective is to predict whether an individual will default on his credit card payment, on the basis of his bank balanc, income and whether he is a student or not

# Dataset can be downloaded from <a href="https://drive.google.com/file/d/1ZFBik4gOMkSydH7FX54KKf9OKtwll7Mw/view?usp=sharing"> here</a>.
A simulated data set containing information on ten thousand customers. 



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
!pip install xlrd
org_data = pd.read_excel("LogisticRegressionDemo.xlsx")
```

    Requirement already satisfied: xlrd in c:\users\i330087\appdata\local\continuum\anaconda3\lib\site-packages (1.2.0)
    


```python
#org_data = org_data.drop(['Unnamed: 0'], axis=1)
org_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>729.526495</td>
      <td>44361.625074</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>No</td>
      <td>Yes</td>
      <td>817.180407</td>
      <td>12106.134700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>No</td>
      <td>No</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>No</td>
      <td>No</td>
      <td>529.250605</td>
      <td>35704.493935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>No</td>
      <td>No</td>
      <td>785.655883</td>
      <td>38463.495879</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the data set dimensions and obtain basic information. <br> shape attribute provides data size i.e. no of obs as rows and no. of features as columns.<br>
# describe function provides statistical summary
print(org_data.shape)
org_data = org_data.drop(['Unnamed: 0'], axis=1)
org_data.info()
# Try org_data.describe() for printing the statistical summary
```

    (10000, 5)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 4 columns):
    default    10000 non-null object
    student    10000 non-null object
    balance    10000 non-null float64
    income     10000 non-null float64
    dtypes: float64(2), object(2)
    memory usage: 312.6+ KB
    

Notice that there are no missing values in the dataset. All columns have non-null values

Print the number of positive and negative instances


```python
print(org_data.default.value_counts())
```

    No     9667
    Yes     333
    Name: default, dtype: int64
    

A heavily class imbalanced problem with only 3.5% non-defaulters


```python
data = org_data[['default',"student",'balance','income']]
data = data[1:1000]
print(data.shape)
data.default.value_counts()
#data.head()
```

    (999, 4)
    




    No     961
    Yes     38
    Name: default, dtype: int64



Since the response variable "default" and the predictor student have categorical values in the form of "yes" and "no", it is important to encode them using 0/1 form to feed as input to the model. <br>
<b> factorize() </b> method is used to perform this conversion


```python
data['default'],default_unique = data['default'].factorize()
data['student'], st_unique = data['student'].factorize()
print("unique student labels",st_unique)
print("unique default labels",default_unique)
data.head()
```

    unique student labels Index(['Yes', 'No'], dtype='object')
    unique default labels Index(['No', 'Yes'], dtype='object')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>817.180407</td>
      <td>12106.134700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>529.250605</td>
      <td>35704.493935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>785.655883</td>
      <td>38463.495879</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>919.588530</td>
      <td>7491.558572</td>
    </tr>
  </tbody>
</table>
</div>



This implies that student='Yes' is encoded as 0 and student='No' is encoded as 1. Similarly, default='No' is encoded as 0 and default='yes' as 1.


```python
data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>default</th>
      <td>1.000000</td>
      <td>-0.018135</td>
      <td>0.360888</td>
      <td>0.014253</td>
    </tr>
    <tr>
      <th>student</th>
      <td>-0.018135</td>
      <td>1.000000</td>
      <td>-0.227472</td>
      <td>0.755412</td>
    </tr>
    <tr>
      <th>balance</th>
      <td>0.360888</td>
      <td>-0.227472</td>
      <td>1.000000</td>
      <td>-0.137466</td>
    </tr>
    <tr>
      <th>income</th>
      <td>0.014253</td>
      <td>0.755412</td>
      <td>-0.137466</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



the highest correlation between default and the other predictors is with Balance. Thus, balance has strong infulence on whether the candidate will default on his credit payment or not. The correlation beween student and income is much lesser.

student has negative correlation with default i.e. chances of a student defaulting is lesser than that of a non-student defaulting.
Also notice the high positive correlation between student and income, negative correlation between student and balance.

It will be good to explore this for different subsets and then the entire dataset to notice the change in these correlation coefficients and observe true relatinship

#### Visualize relationship between response vs predictors

The response variable for a binary linear regression problem has only two options: 0/1, creating a scatter plot is not helpful here. Instead, box plots and contingency tables provide good understanding of the data.


```python
pd.crosstab(data.student, data.default).plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x255e04307b8>



<img src="{{site.url}}{{site.baseurl}}/images/20191214LogReg/output_18_1.png" alt="LogisticRegression">


Is student a good indicator of defaulter?
For defaulter = NO, i.e. non-deafulters, the difference between student=yes(0) and non-student is very prominent i.e. chances of student defaulting is much less than the chances of non-students defaulting. But for defaulters, the distinction is not that significant.


```python
# boxplot
data.boxplot(column='balance',by='default')

```




    <matplotlib.axes._subplots.AxesSubplot at 0x255e069c9b0>




<img src="{{site.url}}{{site.baseurl}}/images/20191214LogReg/output_20_1.png" alt="LogisticRegression">


The purpose of visualization is to think about questions such as: What does this bar chart or box plot convey? <br>
There is a huge difference in balance between defaulters and non-defaulters.
Check for income vs default as well.

Another visualization style in practice is to see relationship between predictors and response using coloured scatter plots as follows


```python
plt.scatter(data.balance, data.income, c=data.default)
```




    <matplotlib.collections.PathCollection at 0x255e05ac940>




<img src="{{site.url}}{{site.baseurl}}/images/20191214LogReg/output_23_1.png" alt="LogisticRegression">



```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```


```python
model1 = LogisticRegression()
# X = data.balance.values.reshape(-1,1)
#X = data[['balance','student']]
X = data[['balance','student','income']]
# print(X.shape)
y = data.default.values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(X,data.default,test_size=0.20,random_state=105)
model1.fit(X, y)
```

    C:\Users\I330087\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\I330087\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
print('classes: ',model1.classes_)
print('coefficients: ',model1.coef_)
print('intercept :', model1.intercept_)
```

    classes:  [0 1]
    coefficients:  [[ 2.30714566e-04  3.36201089e-07 -1.09177044e-04]]
    intercept : [-1.43338127e-06]
    

<b> Model Equation </b>
\begin{equation*}
z = 2.30714566e-04*balance + 3.36201089e^{-07} * student - 1.091e^{-04}*income
\end{equation*}

\begin{equation*}
sigma(z) = \frac{1}{ 1+e^{-z}}
\end{equation*}


```python
# test1={'balance':40, 'student':1,'income':1500}
# x_test = pd.DataFrame(test1, index=[0])
predictions = model1.predict(x_test)
#print(predictions)
score = accuracy_score(predictions, y_test)
print(score)
```

    0.945
    


```python
from sklearn import metrics
print("prediction shape",predictions.shape)
print("y_test.shape", y_test.shape)
cm = metrics.confusion_matrix(predictions, y_test)
print(cm)

```

    prediction shape (200,)
    y_test.shape (200,)
    [[189  11]
     [  0   0]]
    


```python
# Predicting for an unknown test case
test1={'balance':3000, 'student':1,'income':1500}
x_test1 = pd.DataFrame(test1, index=[0])
predictions = model1.predict(x_test1)
print(predictions)

prob = model1.predict_proba(x_test1)
print(prob)
```

    [1]
    [[0.3708955 0.6291045]]
    

How do you interpret the result? 

The sklearn model doesnot provide statistical summary to help us understand which features are not relevant and can be removed.<br> Statsmodel provides a good alternative to this and provides a detailed statistical summary that helps in feature selection as well.


```python
model2 = LogisticRegression()
X2 = data[['student']]
y2 = data[['default']]
model2.fit(X2, y2)
```

    C:\Users\I330087\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\I330087\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\utils\validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
print(model2.coef_)
print(model2.intercept_)
test2 = {'student':1}
x_test2 = pd.DataFrame(test2, index=[0])
model2.predict(x_test2)
prob = model2.predict_proba(x_test2)
print(prob)
```

    [[-0.37162045]]
    [-2.90741904]
    [[0.9637027 0.0362973]]
    

Contrasting results: model 1 predicts the prob of a studen defaulting is higher than non-defaulting whereas model 2 predicts the chance ofa student defaulting is much higehr.


```python
from statsmodels.api import OLS
OLS(data.default,X).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>default</td>     <th>  R-squared (uncentered):</th>      <td>   0.138</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.135</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   52.98</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 14 Dec 2019</td> <th>  Prob (F-statistic):</th>          <td>8.92e-32</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:32:04</td>     <th>  Log-Likelihood:    </th>          <td>  289.39</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   999</td>      <th>  AIC:               </th>          <td>  -572.8</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   996</td>      <th>  BIC:               </th>          <td>  -558.1</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>              <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>balance</th> <td>    0.0001</td> <td>    1e-05</td> <td>   10.750</td> <td> 0.000</td> <td> 8.82e-05</td> <td>    0.000</td>
</tr>
<tr>
  <th>student</th> <td>    0.0215</td> <td>    0.019</td> <td>    1.103</td> <td> 0.270</td> <td>   -0.017</td> <td>    0.060</td>
</tr>
<tr>
  <th>income</th>  <td>-1.666e-06</td> <td> 5.38e-07</td> <td>   -3.095</td> <td> 0.002</td> <td>-2.72e-06</td> <td> -6.1e-07</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>878.203</td> <th>  Durbin-Watson:     </th> <td>   1.838</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>16270.196</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 4.239</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>20.861</td>  <th>  Cond. No.          </th> <td>1.22e+05</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.22e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



<b> NOTE </b>
Remember we had observed high correlation between student and income.
Important to notice the p-values for each of the coefficients. The t-statistics of student [P>|t|] is a very high value and we need to accept the hypothesis tht student has no relation with default and this is one feature that can be removed from consideration.


```python
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
```


```python
gnb_model.fit(x_train, y_train)

```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
gnb_predictions = gnb_model.predict(x_test)
#print(gnb_predictions.shape)
#print(y_test.shape)
score = accuracy_score(gnb_predictions, y_test)
print("Gaussian NB accuracy : ", score)
```

    Gaussian NB accuracy :  0.96
    


```python
gnb_model.class_prior_
```




    array([0.96620776, 0.03379224])



For the current test case, the probability of defaulter (class=0) is much higher than that of a non-defaulter.

<b> NOTE: </b>
For the current subset of data, performance of GNB is better than that of Logistic regression because the training data set is of limited size. 
Try testing it on the whole dataset and then compare the performance.


```python

```
