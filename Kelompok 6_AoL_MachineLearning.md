# AOL Machine Learning

Kelompok 6

Feta Kalih Wigati - 2502019451

Sekar Alisha Firdaus - 2501970531

Salsa Deswina Raihani - 2502069361

Dataset : Diabetes

## Importing Libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import fbeta_score, cohen_kappa_score

SEED = 42
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```


```python
diabet = pd.read_csv('diabetes.csv') 
diabet.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
diabet.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    


```python
diabet.describe()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
diabet = diabet.drop_duplicates()
```

## Feature Engineering

#### Imputation


```python
diabet.isnull().sum()
```




    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64




```python
diabetcopy = diabet.copy(deep = True)
diabetcopy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetcopy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
```


```python
print(diabetcopy.isnull().sum())
```

    Pregnancies                   0
    Glucose                       5
    BloodPressure                35
    SkinThickness               227
    Insulin                     374
    BMI                          11
    DiabetesPedigreeFunction      0
    Age                           0
    Outcome                       0
    dtype: int64
    


```python
# Mengecek apakah ada nilai 0 lewat visualization
p = diabet.hist(color='green',figsize = (10,10))
```


    
![png](output_14_0.png)
    



```python
diabetcopy['Glucose'].fillna(diabetcopy['Glucose'].mean(), inplace = True)
diabetcopy['BloodPressure'].fillna(diabetcopy['BloodPressure'].mean(), inplace = True)
diabetcopy['SkinThickness'].fillna(diabetcopy['SkinThickness'].median(), inplace = True)
diabetcopy['Insulin'].fillna(diabetcopy['Insulin'].median(), inplace = True)
diabetcopy['BMI'].fillna(diabetcopy['BMI'].median(), inplace = True)
```


```python
p = diabetcopy.hist(color='green',figsize = (10,10))
```


    
![png](output_16_0.png)
    


#### Menghitung Value


```python
diabetcopy.count()
```




    Pregnancies                 768
    Glucose                     768
    BloodPressure               768
    SkinThickness               768
    Insulin                     768
    BMI                         768
    DiabetesPedigreeFunction    768
    Age                         768
    Outcome                     768
    dtype: int64




```python
diabet.Outcome.value_counts()
diabet['Outcome'].value_counts().plot(kind='bar').set_title('Diabetes Outcome')
```




    Text(0.5, 1.0, 'Diabetes Outcome')




    
![png](output_19_1.png)
    



```python
diabet.Outcome.value_counts()
```




    0    500
    1    268
    Name: Outcome, dtype: int64



#### Korelasi antar kolom


```python
plt.figure(figsize=(13,10))
sns.heatmap(diabetcopy.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
```




    <AxesSubplot:>




    
![png](output_22_1.png)
    


#### Mencari Outliers

#### Pregnancies Outliers


```python
plt.boxplot(diabetcopy['Pregnancies'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df2665d7f0>,
      <matplotlib.lines.Line2D at 0x1df2665dac0>],
     'caps': [<matplotlib.lines.Line2D at 0x1df2665dd90>,
      <matplotlib.lines.Line2D at 0x1df2666b0a0>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df2665d520>],
     'medians': [<matplotlib.lines.Line2D at 0x1df2666b370>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df2666b640>],
     'means': []}




    
![png](output_25_1.png)
    



```python
q1 =diabetcopy['Pregnancies'].quantile(0.25)
q3 = diabetcopy['Pregnancies'].quantile(0.75)
iqr = q3-q1

lower = q1-1.5*iqr
upper = q3+1.5*iqr

outlier_low = (diabetcopy['Pregnancies']<lower)
outlier_upp = (diabetcopy['Pregnancies']>upper)
diabetcopy['Pregnancies'][(outlier_low|outlier_upp)]
diabetcopy['Pregnancies'][~(outlier_low|outlier_upp)]
diabetcopy = diabetcopy[~(outlier_low|outlier_upp)]
diabetcopy
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>125.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>10</td>
      <td>101.0</td>
      <td>76.0</td>
      <td>48.0</td>
      <td>180.0</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122.0</td>
      <td>70.0</td>
      <td>27.0</td>
      <td>125.0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121.0</td>
      <td>72.0</td>
      <td>23.0</td>
      <td>112.0</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126.0</td>
      <td>60.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93.0</td>
      <td>70.0</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>764 rows × 9 columns</p>
</div>




```python
plt.boxplot(diabetcopy['Pregnancies'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df260cd340>,
      <matplotlib.lines.Line2D at 0x1df260cd610>],
     'caps': [<matplotlib.lines.Line2D at 0x1df260cd8e0>,
      <matplotlib.lines.Line2D at 0x1df260cdbb0>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df260cd070>],
     'medians': [<matplotlib.lines.Line2D at 0x1df260cde80>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df260d9190>],
     'means': []}




    
![png](output_27_1.png)
    


#### Glucose Outliers


```python
plt.boxplot(diabetcopy['Glucose'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df2626c970>,
      <matplotlib.lines.Line2D at 0x1df2626cc40>],
     'caps': [<matplotlib.lines.Line2D at 0x1df2626cf10>,
      <matplotlib.lines.Line2D at 0x1df260fa220>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df2626c6a0>],
     'medians': [<matplotlib.lines.Line2D at 0x1df260fa4f0>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df260fa7c0>],
     'means': []}




    
![png](output_29_1.png)
    


#### BloodPressure Outliers


```python
plt.boxplot(diabetcopy['BloodPressure'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df26152ca0>,
      <matplotlib.lines.Line2D at 0x1df26152f70>],
     'caps': [<matplotlib.lines.Line2D at 0x1df26164280>,
      <matplotlib.lines.Line2D at 0x1df26164550>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df261529d0>],
     'medians': [<matplotlib.lines.Line2D at 0x1df26164820>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df26164af0>],
     'means': []}




    
![png](output_31_1.png)
    



```python
q1 =diabetcopy['BloodPressure'].quantile(0.25)
q3 = diabetcopy['BloodPressure'].quantile(0.75)
iqr = q3-q1

lower = q1-1.5*iqr
upper = q3+1.5*iqr

outlier_low = (diabetcopy['BloodPressure']<lower)
outlier_upp = (diabetcopy['BloodPressure']>upper)

```


```python
diabetcopy['BloodPressure'][(outlier_low|outlier_upp)]

```




    18      30.0
    43     110.0
    84     108.0
    106    122.0
    125     30.0
    177    110.0
    362    108.0
    549    110.0
    597     24.0
    599     38.0
    658    106.0
    662    106.0
    672    106.0
    691    114.0
    Name: BloodPressure, dtype: float64




```python
diabetcopy['BloodPressure'][~(outlier_low|outlier_upp)]

```




    0      72.0
    1      66.0
    2      64.0
    3      66.0
    4      40.0
           ... 
    763    76.0
    764    70.0
    765    72.0
    766    60.0
    767    70.0
    Name: BloodPressure, Length: 750, dtype: float64




```python
diabetcopy = diabetcopy[~(outlier_low|outlier_upp)]
diabetcopy
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>125.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>10</td>
      <td>101.0</td>
      <td>76.0</td>
      <td>48.0</td>
      <td>180.0</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122.0</td>
      <td>70.0</td>
      <td>27.0</td>
      <td>125.0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121.0</td>
      <td>72.0</td>
      <td>23.0</td>
      <td>112.0</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126.0</td>
      <td>60.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93.0</td>
      <td>70.0</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>750 rows × 9 columns</p>
</div>




```python
plt.boxplot(diabetcopy['BloodPressure'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df261c90d0>,
      <matplotlib.lines.Line2D at 0x1df261c93a0>],
     'caps': [<matplotlib.lines.Line2D at 0x1df261c9670>,
      <matplotlib.lines.Line2D at 0x1df261c9940>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df261bbdc0>],
     'medians': [<matplotlib.lines.Line2D at 0x1df261c9c10>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df261c9ee0>],
     'means': []}




    
![png](output_36_1.png)
    


#### SkinThickness Outliers


```python
plt.boxplot(diabetcopy['SkinThickness'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df26226a30>,
      <matplotlib.lines.Line2D at 0x1df26226d00>],
     'caps': [<matplotlib.lines.Line2D at 0x1df26226fd0>,
      <matplotlib.lines.Line2D at 0x1df262742e0>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df26226760>],
     'medians': [<matplotlib.lines.Line2D at 0x1df262745b0>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df26274880>],
     'means': []}




    
![png](output_38_1.png)
    



```python
q1 =diabetcopy['SkinThickness'].quantile(0.25)
q3 = diabetcopy['SkinThickness'].quantile(0.75)
iqr = q3-q1

lower = q1-1.5*iqr
upper = q3+1.5*iqr

outlier_low = (diabetcopy['SkinThickness']<lower)
outlier_upp = (diabetcopy['SkinThickness']>upper)

```


```python
diabetcopy['SkinThickness'][(outlier_low|outlier_upp)]
```




    8      45.0
    16     47.0
    32     11.0
    39     47.0
    50     11.0
           ... 
    698    11.0
    710    13.0
    718    46.0
    753    44.0
    763    48.0
    Name: SkinThickness, Length: 85, dtype: float64




```python
diabetcopy['SkinThickness'][~(outlier_low|outlier_upp)]
```




    0      35.0
    1      29.0
    2      29.0
    3      23.0
    4      35.0
           ... 
    762    29.0
    764    27.0
    765    23.0
    766    29.0
    767    31.0
    Name: SkinThickness, Length: 665, dtype: float64




```python
diabetcopy = diabetcopy[~(outlier_low|outlier_upp)]
diabetcopy
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>125.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>762</th>
      <td>9</td>
      <td>89.0</td>
      <td>62.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>22.5</td>
      <td>0.142</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122.0</td>
      <td>70.0</td>
      <td>27.0</td>
      <td>125.0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121.0</td>
      <td>72.0</td>
      <td>23.0</td>
      <td>112.0</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126.0</td>
      <td>60.0</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93.0</td>
      <td>70.0</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>665 rows × 9 columns</p>
</div>




```python
plt.boxplot(diabetcopy['SkinThickness'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df262e2400>,
      <matplotlib.lines.Line2D at 0x1df262e26d0>],
     'caps': [<matplotlib.lines.Line2D at 0x1df262e29a0>,
      <matplotlib.lines.Line2D at 0x1df262e2c70>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df262e2130>],
     'medians': [<matplotlib.lines.Line2D at 0x1df262e2f40>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df262ed250>],
     'means': []}




    
![png](output_43_1.png)
    


#### Insulin Outliers


```python
plt.boxplot(diabetcopy['Insulin'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df26343730>,
      <matplotlib.lines.Line2D at 0x1df26343a00>],
     'caps': [<matplotlib.lines.Line2D at 0x1df26343cd0>,
      <matplotlib.lines.Line2D at 0x1df26343fa0>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df26343460>],
     'medians': [<matplotlib.lines.Line2D at 0x1df263502b0>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df26350580>],
     'means': []}




    
![png](output_45_1.png)
    



```python
q1 =diabetcopy['Insulin'].quantile(0.25)
q3 = diabetcopy['Insulin'].quantile(0.75)
iqr = q3-q1

lower = q1-1.5*iqr
upper = q3+1.5*iqr

outlier_low = (diabetcopy['Insulin']<lower)
outlier_upp = (diabetcopy['Insulin']>upper)

```


```python
diabetcopy['Insulin'][(outlier_low|outlier_upp)]
```




    3       94.0
    4      168.0
    6       88.0
    13     846.0
    14     175.0
           ...  
    748    200.0
    751     74.0
    755    110.0
    760     16.0
    765    112.0
    Name: Insulin, Length: 307, dtype: float64




```python
diabetcopy['Insulin'][~(outlier_low|outlier_upp)]
```




    0      125.0
    1      125.0
    2      125.0
    5      125.0
    7      125.0
           ...  
    761    125.0
    762    125.0
    764    125.0
    766    125.0
    767    125.0
    Name: Insulin, Length: 358, dtype: float64




```python
diabetcopy = diabetcopy[~(outlier_low|outlier_upp)]
diabetcopy
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.000000</td>
      <td>35.0</td>
      <td>125.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116.0</td>
      <td>74.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>115.0</td>
      <td>72.405184</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>35.3</td>
      <td>0.134</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>761</th>
      <td>9</td>
      <td>170.0</td>
      <td>74.000000</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>44.0</td>
      <td>0.403</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>762</th>
      <td>9</td>
      <td>89.0</td>
      <td>62.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>22.5</td>
      <td>0.142</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122.0</td>
      <td>70.000000</td>
      <td>27.0</td>
      <td>125.0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126.0</td>
      <td>60.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93.0</td>
      <td>70.000000</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>358 rows × 9 columns</p>
</div>




```python
plt.boxplot(diabetcopy['Insulin'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df263a0e20>,
      <matplotlib.lines.Line2D at 0x1df263b1130>],
     'caps': [<matplotlib.lines.Line2D at 0x1df263b1400>,
      <matplotlib.lines.Line2D at 0x1df263b16d0>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df263a0b50>],
     'medians': [<matplotlib.lines.Line2D at 0x1df263b19a0>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df263b1c70>],
     'means': []}




    
![png](output_50_1.png)
    


#### BMI Outliers


```python
plt.boxplot(diabetcopy['BMI'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df26400fa0>,
      <matplotlib.lines.Line2D at 0x1df2640f2b0>],
     'caps': [<matplotlib.lines.Line2D at 0x1df2640f580>,
      <matplotlib.lines.Line2D at 0x1df2640f850>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df26400ca0>],
     'medians': [<matplotlib.lines.Line2D at 0x1df2640fb20>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df2640fe50>],
     'means': []}




    
![png](output_52_1.png)
    



```python
q1 =diabetcopy['BMI'].quantile(0.25)
q3 = diabetcopy['BMI'].quantile(0.75)
iqr = q3-q1

lower = q1-1.5*iqr
upper = q3+1.5*iqr

outlier_low = (diabetcopy['BMI']<lower)
outlier_upp = (diabetcopy['BMI']>upper)
diabetcopy['BMI'][(outlier_low|outlier_upp)]
diabetcopy['BMI'][~(outlier_low|outlier_upp)]
diabetcopy = diabetcopy[~(outlier_low|outlier_upp)]
diabetcopy
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.000000</td>
      <td>35.0</td>
      <td>125.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116.0</td>
      <td>74.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>115.0</td>
      <td>72.405184</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>35.3</td>
      <td>0.134</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>761</th>
      <td>9</td>
      <td>170.0</td>
      <td>74.000000</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>44.0</td>
      <td>0.403</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>762</th>
      <td>9</td>
      <td>89.0</td>
      <td>62.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>22.5</td>
      <td>0.142</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122.0</td>
      <td>70.000000</td>
      <td>27.0</td>
      <td>125.0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126.0</td>
      <td>60.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93.0</td>
      <td>70.000000</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>353 rows × 9 columns</p>
</div>




```python
plt.boxplot(diabetcopy['BMI'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df2457f550>,
      <matplotlib.lines.Line2D at 0x1df23fa71f0>],
     'caps': [<matplotlib.lines.Line2D at 0x1df23fa7f10>,
      <matplotlib.lines.Line2D at 0x1df23fa79d0>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df2457fe50>],
     'medians': [<matplotlib.lines.Line2D at 0x1df23fa78b0>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df23fa74f0>],
     'means': []}




    
![png](output_54_1.png)
    


#### DiabetesPedigreeFunction Outliers


```python
plt.boxplot(diabetcopy['DiabetesPedigreeFunction'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df24558af0>,
      <matplotlib.lines.Line2D at 0x1df24558dc0>],
     'caps': [<matplotlib.lines.Line2D at 0x1df24558310>,
      <matplotlib.lines.Line2D at 0x1df24550c10>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df245584f0>],
     'medians': [<matplotlib.lines.Line2D at 0x1df245501c0>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df245507c0>],
     'means': []}




    
![png](output_56_1.png)
    



```python
q1 =diabetcopy['DiabetesPedigreeFunction'].quantile(0.25)
q3 = diabetcopy['DiabetesPedigreeFunction'].quantile(0.75)
iqr = q3-q1

lower = q1-1.5*iqr
upper = q3+1.5*iqr

outlier_low = (diabetcopy['DiabetesPedigreeFunction']<lower)
outlier_upp = (diabetcopy['DiabetesPedigreeFunction']>upper)
diabetcopy['DiabetesPedigreeFunction'][(outlier_low|outlier_upp)]
diabetcopy['DiabetesPedigreeFunction'][~(outlier_low|outlier_upp)]
diabetcopy = diabetcopy[~(outlier_low|outlier_upp)]
diabetcopy
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.000000</td>
      <td>35.0</td>
      <td>125.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116.0</td>
      <td>74.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>115.0</td>
      <td>72.405184</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>35.3</td>
      <td>0.134</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>761</th>
      <td>9</td>
      <td>170.0</td>
      <td>74.000000</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>44.0</td>
      <td>0.403</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>762</th>
      <td>9</td>
      <td>89.0</td>
      <td>62.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>22.5</td>
      <td>0.142</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122.0</td>
      <td>70.000000</td>
      <td>27.0</td>
      <td>125.0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126.0</td>
      <td>60.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93.0</td>
      <td>70.000000</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>336 rows × 9 columns</p>
</div>




```python
plt.boxplot(diabetcopy['DiabetesPedigreeFunction'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df2450f070>,
      <matplotlib.lines.Line2D at 0x1df2450fd60>],
     'caps': [<matplotlib.lines.Line2D at 0x1df2450f430>,
      <matplotlib.lines.Line2D at 0x1df2450f8b0>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df2450f190>],
     'medians': [<matplotlib.lines.Line2D at 0x1df245015b0>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df24501e50>],
     'means': []}




    
![png](output_58_1.png)
    


#### Age Outliers


```python
    plt.boxplot(diabetcopy['Age'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df240c8b80>,
      <matplotlib.lines.Line2D at 0x1df240c84f0>],
     'caps': [<matplotlib.lines.Line2D at 0x1df240c8760>,
      <matplotlib.lines.Line2D at 0x1df245c1190>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df240c8c40>],
     'medians': [<matplotlib.lines.Line2D at 0x1df2459cac0>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df2459c580>],
     'means': []}




    
![png](output_60_1.png)
    



```python
q1 =diabetcopy['Age'].quantile(0.25)
q3 = diabetcopy['Age'].quantile(0.75)
iqr = q3-q1

lower = q1-1.5*iqr
upper = q3+1.5*iqr

outlier_low = (diabetcopy['Age']<lower)
outlier_upp = (diabetcopy['Age']>upper)
diabetcopy['Age'][(outlier_low|outlier_upp)]
diabetcopy['Age'][~(outlier_low|outlier_upp)]
diabetcopy = diabetcopy[~(outlier_low|outlier_upp)]
diabetcopy
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.000000</td>
      <td>35.0</td>
      <td>125.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116.0</td>
      <td>74.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>115.0</td>
      <td>72.405184</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>35.3</td>
      <td>0.134</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>761</th>
      <td>9</td>
      <td>170.0</td>
      <td>74.000000</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>44.0</td>
      <td>0.403</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>762</th>
      <td>9</td>
      <td>89.0</td>
      <td>62.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>22.5</td>
      <td>0.142</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122.0</td>
      <td>70.000000</td>
      <td>27.0</td>
      <td>125.0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126.0</td>
      <td>60.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93.0</td>
      <td>70.000000</td>
      <td>31.0</td>
      <td>125.0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>332 rows × 9 columns</p>
</div>




```python
plt.boxplot(diabetcopy['Age'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df2376b0d0>,
      <matplotlib.lines.Line2D at 0x1df2376b3a0>],
     'caps': [<matplotlib.lines.Line2D at 0x1df2376b670>,
      <matplotlib.lines.Line2D at 0x1df2376b940>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df240dddc0>],
     'medians': [<matplotlib.lines.Line2D at 0x1df2376bc10>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df2376bee0>],
     'means': []}




    
![png](output_62_1.png)
    


#### Outcome Outliers 


```python
    plt.boxplot(diabetcopy['Outcome'])
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x1df24057190>,
      <matplotlib.lines.Line2D at 0x1df24057460>],
     'caps': [<matplotlib.lines.Line2D at 0x1df24057760>,
      <matplotlib.lines.Line2D at 0x1df24057a30>],
     'boxes': [<matplotlib.lines.Line2D at 0x1df23fe0e80>],
     'medians': [<matplotlib.lines.Line2D at 0x1df24057d00>],
     'fliers': [<matplotlib.lines.Line2D at 0x1df24057fd0>],
     'means': []}




    
![png](output_64_1.png)
    



```python
diabetcopy.shape
```




    (332, 9)



#### Scaling


```python
 diabetcopy.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.000000</td>
      <td>35.0</td>
      <td>125.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116.0</td>
      <td>74.000000</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>115.0</td>
      <td>72.405184</td>
      <td>29.0</td>
      <td>125.0</td>
      <td>35.3</td>
      <td>0.134</td>
      <td>29</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetcopy.drop(["Outcome"],axis = 1),), columns=['Pregnancies', 
'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.490124</td>
      <td>1.001053</td>
      <td>-0.157383</td>
      <td>1.439678</td>
      <td>0.0</td>
      <td>0.366217</td>
      <td>1.204203</td>
      <td>1.186805</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.005476</td>
      <td>-1.160124</td>
      <td>-0.745165</td>
      <td>0.015726</td>
      <td>0.0</td>
      <td>-0.790311</td>
      <td>-0.077565</td>
      <td>-0.355186</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.088364</td>
      <td>2.201707</td>
      <td>-0.941092</td>
      <td>0.015726</td>
      <td>0.0</td>
      <td>-1.335532</td>
      <td>1.413186</td>
      <td>-0.274028</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.191004</td>
      <td>-0.096687</td>
      <td>0.038545</td>
      <td>0.015726</td>
      <td>0.0</td>
      <td>-0.955530</td>
      <td>-0.774177</td>
      <td>-0.436343</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.686604</td>
      <td>-0.130992</td>
      <td>-0.117689</td>
      <td>0.015726</td>
      <td>0.0</td>
      <td>0.647089</td>
      <td>-1.085331</td>
      <td>-0.517501</td>
    </tr>
  </tbody>
</table>
</div>



## Model Building


```python
X = diabetcopy.drop('Outcome', axis=1)
y = diabetcopy['Outcome']

# split data to 80:20 ratio for train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y) 
```

### Random Forest


```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
```




    RandomForestClassifier(n_estimators=200)




```python
#Check the accuracy score for random forest
from sklearn import metrics

predictions = rfc.predict(X_test)
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, predictions)))
```

    Accuracy_Score = 0.7910447761194029
    


```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
```

    [[40  4]
     [10 13]]
                  precision    recall  f1-score   support
    
               0       0.80      0.91      0.85        44
               1       0.76      0.57      0.65        23
    
        accuracy                           0.79        67
       macro avg       0.78      0.74      0.75        67
    weighted avg       0.79      0.79      0.78        67
    
    

### Support Vector Machine (SVM)


```python
from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
```




    SVC()




```python
svc_pred = svc_model.predict(X_test)
```


```python
from sklearn import metrics

print("Accuracy Score =", format(metrics.accuracy_score(y_test, svc_pred)))
```

    Accuracy Score = 0.7313432835820896
    


```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test,svc_pred))
```

    [[43  1]
     [17  6]]
                  precision    recall  f1-score   support
    
               0       0.72      0.98      0.83        44
               1       0.86      0.26      0.40        23
    
        accuracy                           0.73        67
       macro avg       0.79      0.62      0.61        67
    weighted avg       0.76      0.73      0.68        67
    
    

### Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
```




    DecisionTreeClassifier()




```python
from sklearn import metrics

predictions = dtree.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test,predictions)))
```

    Accuracy Score = 0.6716417910447762
    


```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
```

    [[32 12]
     [10 13]]
                  precision    recall  f1-score   support
    
               0       0.76      0.73      0.74        44
               1       0.52      0.57      0.54        23
    
        accuracy                           0.67        67
       macro avg       0.64      0.65      0.64        67
    weighted avg       0.68      0.67      0.67        67
    
    

### Gaussian Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
```




    GaussianNB()




```python
from sklearn import metrics

predictions = gnb.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test,predictions)))
```

    Accuracy Score = 0.8059701492537313
    


```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
```

    [[39  5]
     [ 8 15]]
                  precision    recall  f1-score   support
    
               0       0.83      0.89      0.86        44
               1       0.75      0.65      0.70        23
    
        accuracy                           0.81        67
       macro avg       0.79      0.77      0.78        67
    weighted avg       0.80      0.81      0.80        67
    
    

### Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
```




    LogisticRegression()




```python
from sklearn import metrics

predictions = logreg.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test,predictions)))
```

    Accuracy Score = 0.8208955223880597
    


```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
```

    [[43  1]
     [11 12]]
                  precision    recall  f1-score   support
    
               0       0.80      0.98      0.88        44
               1       0.92      0.52      0.67        23
    
        accuracy                           0.82        67
       macro avg       0.86      0.75      0.77        67
    weighted avg       0.84      0.82      0.81        67
    
    

## The Conclusion From Model

We have done five model building, from the five models we have done we can conclude the logistic regression model is the best model because it has the highest accuracy value among the five existing models, with an accuracy value of 0.82

## Saving Model (Logistic Regression)


```python
import pickle
# save the model using pickle
savedmodel = pickle.dumps(logreg)
```


```python
# loading that saved model
logreg_from_pickle = pickle.loads(savedmodel)
```


```python
# make predictions
logreg_from_pickle.predict(X_test)
```




    array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
           0], dtype=int64)



The model has been saved, we can then examine a random feature set of the heads and tails of the data to test whether our model is good enough to provide correct predictions.


```python
diabet.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
diabet.tail()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>763</th>
      <td>10</td>
      <td>101</td>
      <td>76</td>
      <td>48</td>
      <td>180</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121</td>
      <td>72</td>
      <td>23</td>
      <td>112</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93</td>
      <td>70</td>
      <td>31</td>
      <td>0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# make predictions based on patient no 4
logreg.predict([[0,137,40,35,168,43.1,2.228,33]]) 
```




    array([1], dtype=int64)



Patient is predicted to have diabetes. 


```python
# make predictions based on patient no 763
logreg.predict([[10,101,76,48,180,32.9,0.171,63]])
```




    array([0], dtype=int64)



Patient is predicted not to have diabetes. 

The results are in accordance with the predictions, which means the model is proven to have a good level of prediction.

## Conclusion

In this diabetes dataset with all existing patient records, we can build a machine learning model to predict whether the patient in the data set has diabetes or not, we use five models and we can conclude that the logistic regression model is the best model with the highest accuracy value which is 0.82, along with that we can also draw some insights from the data through data analysis and visualization.
