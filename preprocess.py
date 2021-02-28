import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy

# import data
def import_data():
    data = pd.read_csv("dataset/train.csv")
  #  print(data)
    return data

# extract data
# independent variables
def extract_data(data):
    x = data.iloc[:, :-1].values
    # print("Data: \n" + str(x))
    # dependent variable
    y = data.iloc[:, -1].values
    # print("Output: \n" + str(y))
    return x, y

# check missing values
def check_missing_vals(data):
    # check for missing values
    missing_vals = data.isna().sum()
    print("Missing values: \n" + str(missing_vals))
    return missing_vals.sum() != 0

def check_count_of_zero(data, column):
    return data[column].isin([0]).sum()

def check_count_of_value(data, column, value):
    return data[column].isin([value]).sum()

def replace_zero_vales_with_median(data):
    columns=["px_height", "sc_w"]
    for column in columns:
        median_value = data[column].median()
        data = data.replace({column: {0: median_value}}) 

    return data

# split dataset
def split_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


# feature scaling
def scale_features(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test

def preprocess_data(scale=True):
    data = import_data()
    #data_head = data.head()
    #print(data.columns)

    data = replace_zero_vales_with_median(data)

    x, y = extract_data(data)

    x_train, x_test, y_train, y_test = split_dataset(x, y)

    if scale:
        x_train, x_test = scale_features(x_train, x_test)

    return x_train, x_test, y_train, y_test




"""
df1=data.iloc[:, 0:10]
df2=data.iloc[:, 10:20]

print(df1.describe(include="all"))
print(df2.describe(include="all"))

print(check_count_of_zero(data, 'px_height'))
print(check_count_of_zero(data, 'sc_w'))

print(check_count_of_value(data, 'px_height', data['px_height'].median()))
print(check_count_of_value(data, 'sc_w', data['sc_w'].median()))

df1=data.iloc[:, 0:10]
df2=data.iloc[:, 10:20]

print(df1.describe(include="all"))
print(df2.describe(include="all"))

print(check_count_of_zero(data, 'px_height'))
print(check_count_of_zero(data, 'sc_w'))

print(check_count_of_value(data, 'px_height', data['px_height'].median()))
print(check_count_of_value(data, 'sc_w', data['sc_w'].median()))
"""