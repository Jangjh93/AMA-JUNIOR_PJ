```python
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.impute import SimpleImputer

```


```python
rawdata = pd.read_csv('C:/Users/Juhyeon/Documents/Juhyeon/hotel_booking/hotel_bookings.csv')
```


```python
# data exploration
```


```python
rawdata.describe()
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
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>booking_changes</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119386.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>103050.000000</td>
      <td>6797.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.370416</td>
      <td>104.011416</td>
      <td>2016.156554</td>
      <td>27.165173</td>
      <td>15.798241</td>
      <td>0.927599</td>
      <td>2.500302</td>
      <td>1.856403</td>
      <td>0.103890</td>
      <td>0.007949</td>
      <td>0.031912</td>
      <td>0.087118</td>
      <td>0.137097</td>
      <td>0.221124</td>
      <td>86.693382</td>
      <td>189.266735</td>
      <td>2.321149</td>
      <td>101.831122</td>
      <td>0.062518</td>
      <td>0.571363</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.482918</td>
      <td>106.863097</td>
      <td>0.707476</td>
      <td>13.605138</td>
      <td>8.780829</td>
      <td>0.998613</td>
      <td>1.908286</td>
      <td>0.579261</td>
      <td>0.398561</td>
      <td>0.097436</td>
      <td>0.175767</td>
      <td>0.844336</td>
      <td>1.497437</td>
      <td>0.652306</td>
      <td>110.774548</td>
      <td>131.655015</td>
      <td>17.594721</td>
      <td>50.535790</td>
      <td>0.245291</td>
      <td>0.792798</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>-6.380000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>2016.000000</td>
      <td>16.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>69.290000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.000000</td>
      <td>69.000000</td>
      <td>2016.000000</td>
      <td>28.000000</td>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>179.000000</td>
      <td>0.000000</td>
      <td>94.575000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1.000000</td>
      <td>160.000000</td>
      <td>2017.000000</td>
      <td>38.000000</td>
      <td>23.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>229.000000</td>
      <td>270.000000</td>
      <td>0.000000</td>
      <td>126.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.000000</td>
      <td>737.000000</td>
      <td>2017.000000</td>
      <td>53.000000</td>
      <td>31.000000</td>
      <td>19.000000</td>
      <td>50.000000</td>
      <td>55.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>26.000000</td>
      <td>72.000000</td>
      <td>21.000000</td>
      <td>535.000000</td>
      <td>543.000000</td>
      <td>391.000000</td>
      <td>5400.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.set_option("display.max_columns", None)
rawdata.head()
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>market_segment</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Online TA</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>2015-07-03</td>
    </tr>
  </tbody>
</table>
</div>




```python
rawdata.groupby(['deposit_type']).size().reset_index(name='counts')
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
      <th>deposit_type</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>No Deposit</td>
      <td>104641</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Non Refund</td>
      <td>14587</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Refundable</td>
      <td>162</td>
    </tr>
  </tbody>
</table>
</div>




```python
rawdata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 119390 entries, 0 to 119389
    Data columns (total 32 columns):
    hotel                             119390 non-null object
    is_canceled                       119390 non-null int64
    lead_time                         119390 non-null int64
    arrival_date_year                 119390 non-null int64
    arrival_date_month                119390 non-null object
    arrival_date_week_number          119390 non-null int64
    arrival_date_day_of_month         119390 non-null int64
    stays_in_weekend_nights           119390 non-null int64
    stays_in_week_nights              119390 non-null int64
    adults                            119390 non-null int64
    children                          119386 non-null float64
    babies                            119390 non-null int64
    meal                              119390 non-null object
    country                           118902 non-null object
    market_segment                    119390 non-null object
    distribution_channel              119390 non-null object
    is_repeated_guest                 119390 non-null int64
    previous_cancellations            119390 non-null int64
    previous_bookings_not_canceled    119390 non-null int64
    reserved_room_type                119390 non-null object
    assigned_room_type                119390 non-null object
    booking_changes                   119390 non-null int64
    deposit_type                      119390 non-null object
    agent                             103050 non-null float64
    company                           6797 non-null float64
    days_in_waiting_list              119390 non-null int64
    customer_type                     119390 non-null object
    adr                               119390 non-null float64
    required_car_parking_spaces       119390 non-null int64
    total_of_special_requests         119390 non-null int64
    reservation_status                119390 non-null object
    reservation_status_date           119390 non-null object
    dtypes: float64(4), int64(16), object(12)
    memory usage: 29.1+ MB
    


```python
rawdata.isnull().sum().sort_values(ascending = False)
```




    company                           112593
    agent                              16340
    country                              488
    children                               4
    lead_time                              0
    arrival_date_year                      0
    arrival_date_month                     0
    arrival_date_week_number               0
    is_canceled                            0
    market_segment                         0
    arrival_date_day_of_month              0
    stays_in_weekend_nights                0
    stays_in_week_nights                   0
    adults                                 0
    babies                                 0
    meal                                   0
    reservation_status_date                0
    distribution_channel                   0
    reservation_status                     0
    is_repeated_guest                      0
    previous_cancellations                 0
    previous_bookings_not_canceled         0
    reserved_room_type                     0
    assigned_room_type                     0
    booking_changes                        0
    deposit_type                           0
    days_in_waiting_list                   0
    customer_type                          0
    adr                                    0
    required_car_parking_spaces            0
    total_of_special_requests              0
    hotel                                  0
    dtype: int64




```python
# Taking care of missing values

## Drop 'company' column since most of its values are missing
newdata = rawdata.drop(['company'], axis=1)
## Simple imputer for numerical values 
n_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
c_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
n_imputer.fit(newdata.loc[:,['agent','children']])
c_imputer.fit(newdata.loc[:,['country']])
newdata.loc[:,['agent','children']] = n_imputer.transform(newdata.loc[:,['agent','children']])
newdata.loc[:,['country']] = c_imputer.transform(newdata.loc[:,['country']])

```


```python
newdata.isnull().sum().sort_values(ascending = False)
```




    reservation_status_date           0
    market_segment                    0
    is_canceled                       0
    lead_time                         0
    arrival_date_year                 0
    arrival_date_month                0
    arrival_date_week_number          0
    arrival_date_day_of_month         0
    stays_in_weekend_nights           0
    stays_in_week_nights              0
    adults                            0
    children                          0
    babies                            0
    meal                              0
    country                           0
    distribution_channel              0
    reservation_status                0
    is_repeated_guest                 0
    previous_cancellations            0
    previous_bookings_not_canceled    0
    reserved_room_type                0
    assigned_room_type                0
    booking_changes                   0
    deposit_type                      0
    agent                             0
    days_in_waiting_list              0
    customer_type                     0
    adr                               0
    required_car_parking_spaces       0
    total_of_special_requests         0
    hotel                             0
    dtype: int64




```python
# Correcting data types
newdata['children'] = newdata['children'].astype(int)
```




    dtype('int32')




```python
datetime_object = newdata['arrival_date_month'].str[0:3]
month_number = np.zeros(len(datetime_object))

# Creating a new column based on numerical representation of the months
for i in range(0, len(datetime_object)):
    datetime_object[i] = datetime.datetime.strptime(datetime_object[i], "%b")
    month_number[i] = datetime_object[i].month

# Float to integer conversion
month_number = pd.DataFrame(month_number).astype(int)

# 3 columns are merged into one
newdata['arrival_date'] = newdata['arrival_date_year'].map(str) + '-' + month_number[0].map(str) + '-' \
                       + newdata['arrival_date_day_of_month'].map(str)
# Dropping already used columns
newdata = newdata.drop(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month',
                        'arrival_date_week_number'], axis=1)
newdata.head()

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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>market_segment</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
      <th>arrival_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>86.693382</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
      <td>2015-7-1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>86.693382</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
      <td>2015-7-1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>86.693382</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
      <td>2015-7-1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>304.000000</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
      <td>2015-7-1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Online TA</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>240.000000</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>2015-07-03</td>
      <td>2015-7-1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
