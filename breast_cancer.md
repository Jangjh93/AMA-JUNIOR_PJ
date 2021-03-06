```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
origin_dataset = pd.read_csv("C:/Users/Juhyeon/Documents/Juhyeon/breast_cancer/breast-cancer.csv")
dataset = origin_dataset.iloc[:, 1:12]
X = origin_dataset.iloc[:, 2:12].values
y = origin_dataset.iloc[:, 1].values
```


```python
origin_dataset.head()
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
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <td>1</td>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <td>2</td>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <td>3</td>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <td>4</td>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows ?? 32 columns</p>

</div>




```python
dataset.head()
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
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
    </tr>
    <tr>
      <td>1</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
    </tr>
    <tr>
      <td>2</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
    </tr>
    <tr>
      <td>3</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
    </tr>
    <tr>
      <td>4</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
    </tr>
  </tbody>
</table>

</div>




```python
dataset.shape
```




    (569, 11)




```python
pd.DataFrame(dataset).dtypes
```




    diagnosis                  object
    radius_mean               float64
    texture_mean              float64
    perimeter_mean            float64
    area_mean                 float64
    smoothness_mean           float64
    compactness_mean          float64
    concavity_mean            float64
    concave points_mean       float64
    symmetry_mean             float64
    fractal_dimension_mean    float64
    dtype: object




```python
pd.DataFrame(dataset)['diagnosis'].unique()
```




    array(['M', 'B'], dtype=object)




```python
# diagnosis rate
pos = (pd.DataFrame(dataset)['diagnosis']=="M").sum()
neg = (pd.DataFrame(dataset)['diagnosis']=="B").sum()
np.array([['pos', 'neg'],[round(pos/dataset.shape[0],2),round(neg/dataset.shape[0],2)]])
```




    array([['pos', 'neg'],
           ['0.37', '0.63']], dtype='<U4')




```python
# missing value check -> no imputation needed
pd.DataFrame(dataset).isnull().sum().sort_values(ascending = True)
```




    diagnosis                 0
    radius_mean               0
    texture_mean              0
    perimeter_mean            0
    area_mean                 0
    smoothness_mean           0
    compactness_mean          0
    concavity_mean            0
    concave points_mean       0
    symmetry_mean             0
    fractal_dimension_mean    0
    dtype: int64




```python
# Encoding the Dependent Variable (one hot encoding is not needed for independent variables here)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
```

    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     0 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0 1 1 0 1 1 0 0 0 0 1 0 1 1 0 0 0 0 1 0 1 1
     0 1 0 1 1 0 0 0 1 1 0 1 1 1 0 0 0 1 0 0 1 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0
     0 0 0 0 0 0 1 1 1 0 1 1 0 0 0 1 1 0 1 0 1 1 0 1 1 0 0 1 0 0 1 0 0 0 0 1 0
     0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0 1 0 0 1 1 0 0 1 1 0 0 0 0 1 0 0 1 1 1 0 1
     0 1 0 0 0 1 0 0 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 0 0 1 0 1 1 1 1 0 0 1 1 0 0
     0 1 0 0 0 0 0 1 1 0 0 1 0 0 1 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0
     0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 1 1 1 0 0
     0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1
     1 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0
     0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 1 0 0 0 0 0 1 0 0
     1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0
     0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0 1 0 1 0 1 1
     0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 1 1 1 1 1 1 0]



```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
```


```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


```python
# Logistic Regression
np.warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
m1 = LogisticRegression(random_state=0)
m1.fit(X_train,y_train)
```




    LogisticRegression(random_state=0)




```python
# Accuracy, Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = m1.predict(sc.transform(X_test))
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm)
print("Accuracy: %.3f" % acc)
```

    [[90 18]
     [16 47]]
    Accuracy: 0.801



```python
# SVM
from sklearn.svm import SVC
m2 = SVC(random_state=0, kernel='rbf')
m2.fit(X_train, y_train)
```




    SVC(random_state=0)




```python
# Accuracy, Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = m2.predict(sc.transform(X_test))
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm)
print("Accuracy: %.3f" % acc)
```

    [[108   0]
     [ 63   0]]
    Accuracy: 0.632



```python
# KNN
from sklearn.neighbors import KNeighborsClassifier
m3 = KNeighborsClassifier()
m3.fit(X_train,y_train)
```




    KNeighborsClassifier()




```python
# Accuracy, Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = m3.predict(sc.transform(X_test))
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm)
print("Accuracy: %.3f" % acc)
```

    [[106   2]
     [ 38  25]]
    Accuracy: 0.766



```python
# RF
from sklearn.ensemble import RandomForestClassifier
m4 = RandomForestClassifier(n_estimators=10,random_state=0)
m4.fit(X_train, y_train)
```




    RandomForestClassifier(n_estimators=10, random_state=0)




```python
# Accuracy, Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = m4.predict(sc.transform(X_test))
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm)
print("Accuracy: %.3f" % acc)
```

    [[108   0]
     [ 63   0]]
    Accuracy: 0.632



```python
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
m5 = DecisionTreeClassifier(random_state=0, criterion='entropy')
m5.fit(X_train, y_train)
```




    DecisionTreeClassifier(criterion='entropy', random_state=0)




```python
# Accuracy, Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = m5.predict(sc.transform(X_test))
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm)
print("Accuracy: %.3f" % acc)
```

    [[108   0]
     [ 63   0]]
    Accuracy: 0.632



```python
# Automated Machine Learning Using TPOT
import torch
import tpot
from tpot import TPOTClassifier
from sklearn.model_selection import StratifiedKFold
```


```python
# model evaluation definition, 10 fold StratifiedKFold used here
cv = StratifiedKFold(n_splits=10)
# define TPOTClassifier
automl_model = TPOTClassifier(generations=5, population_size=100, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
# performing the search for best fit
automl_model.fit(X, y)
# exporting best model
automl_model.export('tpot_data.py')
```


    HBox(children=(IntProgress(value=0, description='Optimization Progress', max=600, style=ProgressStyle(descript???


???    

    Generation 1 - Current best internal CV score: 0.9525375939849624
    
    Generation 2 - Current best internal CV score: 0.9542919799498746
    
    Generation 3 - Current best internal CV score: 0.9560463659147869
    
    Generation 4 - Current best internal CV score: 0.9613721804511279
    
    Generation 5 - Current best internal CV score: 0.9613721804511279
    
    Best pipeline: GradientBoostingClassifier(RobustScaler(input_matrix), learning_rate=1.0, max_depth=3, max_features=0.9500000000000001, min_samples_leaf=10, min_samples_split=14, n_estimators=100, subsample=1.0)



```python
# TPOT result
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=1)

# Average CV score on the training set was: 0.9613721804511279
exported_pipeline = make_pipeline(
    RobustScaler(),
    GradientBoostingClassifier(learning_rate=1.0, max_depth=3, max_features=0.9500000000000001, min_samples_leaf=10, min_samples_split=14, n_estimators=100, subsample=1.0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

```
