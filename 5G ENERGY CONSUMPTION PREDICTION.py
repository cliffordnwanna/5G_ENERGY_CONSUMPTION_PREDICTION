
"""
 Project Overview :

This project aims to predict energy consumption in 5G base stations using Supervised Learning Regression techniques. The goal is to model and estimate the energy consumed by different 5G base stations based on various features such as load, transmitting power, and energy-saving methods. This is particularly relevant given the increasing cost of energy consumption in telecom operations.

The dataset and challenge were provided by the International Telecommunication Union (ITU) as part of a global competition in 2023. This project uses a subset of the dataset for learning purposes.

Type of Problem: Supervised Regression

Problem Statement:

Network operational expenditure (OPEX) accounts for around 25% of a telecom operator's costs, with 90% of it being energy bills. A significant portion of this energy is consumed by the **Radio Access Network (RAN)**, particularly by **base stations (BSs)**. The goal is to build a machine learning model that can estimate energy consumption based on various network and traffic parameters.

Dataset description : This dataset is derived from the original copy and simplified for learning purposes. It includes cell-level traffic statistics of 4G/5G sites collected on different days.

➡️ Dataset link: https://drive.google.com/file/d/1vW9TA7KAn-OJjD_o9Rd0l6sx77wNaiuk/view

Dataset Key Information :

Time --------- date and time

BS ---------- Base station

Energy --------- Energy consumed

Load ------- total load

ESMODE -------- Energy-saving method

TXpower ---- Transmitting power

TYPE OF PROBLEM : This is a supervised regression machine learning problem
SOLUTION : Multiple linear regression

STEPS:
1. Import you data and perform basic data exploration phase
 - Display general information about the dataset
 - Create a pandas profiling reports to gain insights into the dataset
 - Handle Missing and corrupted values
 - Remove duplicates, if they exist
 - Handle outliers, if they exist
 - Encode categorical features

2. Select your target variable and the features
3. Split your dataset to training and test sets
4. Based on your data exploration phase select a ML regression algorithm and train it on the training set
5. Assess your model performance on the test set using relevant evaluation metrics
6. Discuss alternative ways to improve your model performance
"""


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
from google.colab import drive
drive.mount('/content/drive')

#load dataset
df = pd.read_csv("/content/drive/MyDrive/Untitled folder/DATASETS/5G_energy_consumption_dataset.csv")

# Make a copy of the original dataframe
df1 = df.copy()

df.head()

df.info()

df.describe()
# "Energy" and "ESMODE" columns seem to contain outliers

#Create a pandas profiling reports to gain insights into the dataset

# Installation and importation of libraries
#!pip install ydata-profiling
#from ydata_profiling import ProfileReport


# Create a Pandas Profiling report
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)


# Display and explore the report
profile

"""DATA PREPROCESSING"""

#from df.info() i observed that the datatype of the "Time" column is object.
# I used the pd.to_datetime function to parse the date and time values.
#This function will convert the strings into datetime objects, allowing me to perform various date and time operations
df['Time'] = pd.to_datetime(df['Time'])

# Handle Missing and corrupted values
# Check for missing values
df.isnull().sum()
# There are no missing values in the dataframe

# Remove duplicates, if they exist
df.duplicated().sum()
#df.drop_duplicates()
# The dataframe has no duplicated rows

# Handle outliers
# columns "Energy" and "ESMODE" contain outliers and is postively skewed
# Sort 'Energy' and 'ESMODE' in ascending order to check the outliers
sorted_df = df.sort_values(by=['Energy','ESMODE'], ascending=True)

print("\nDataFrame with 'Column1' sorted in ascending order:")
print(sorted_df)

sorted_df.tail(50)

"""I decided to handle the outliers in the "ESMODE" only becacuse "Energy" is the target column"""

# i created a boxplot to show the outliers in the 'ESMODE' column
# Extract the 'Energy' column from the DataFrame
esmode = df['ESMODE']

# Create a box plot for the 'Energy' column
plt.boxplot(esmode)
plt.title('Box Plot: ESMODE')
plt.ylabel('Esmode')
plt.show()

from scipy import stats

# Define a function to handle outliers using the Z-score method
def handle_outliers_zscore(df, columns, z_threshold=3):
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = df[(z_scores > z_threshold)]
        df = df[(z_scores <= z_threshold)]
        print(f"Outliers removed in column '{col}': {len(outliers)}")
    return df

# Define the columns you want to handle outliers for
columns_to_handle_outliers = ['ESMODE']

# Handle outliers in the specified columns
df_cleaned = handle_outliers_zscore(df, columns_to_handle_outliers)

# df_cleaned now contains the DataFrame with outliers removed

# i will proceed with further analysis and modeling using df_cleaned

# I created a box plot to show the 'ESMODE' column after the outliers have been handled
# Extract the 'Energy' column from the DataFrame
esmode = df_cleaned['ESMODE']

# Create a box plot for the 'Energy' column
plt.boxplot(esmode)
plt.title('Box Plot: ESMODE')
plt.ylabel('Esmode')
plt.show()

# Step 7: Encode categorical features
# contains categorical features,label encoding to convert them into numerical format
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_cleaned['BS'] = encoder.fit_transform(df_cleaned['BS'])

df_cleaned.info()

# Import necessary libraries
from sklearn.model_selection import train_test_split

# Select the target variable and the features
# Target variable = 'Energy' and the features = independent variables(df_cleaned minus "Energy" and "Time")

X = df_cleaned.drop(['Energy','Time'], axis=1)  # Features
target = df_cleaned['Energy']  # Target variable

#machine learning algorithms handle arrays and not dataframes
#therefore, i converted the target variable (y) and features (x) into arrays
y = np.array(target).reshape(-1, 1)

# Split your dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select a regression algorithm and train it on the training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score
# Step 11: Assess your model performance on the test set using relevant evaluation metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2) Score: {r2}")

from sklearn.tree import DecisionTreeRegressor

# Create a Decision Tree Regressor
tree_model = DecisionTreeRegressor()

# Train the model
tree_model.fit(X_train, y_train)

# Predict on the test set
y_pred_tree = tree_model.predict(X_test)

# Calculate MSE and R2
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f"Mean Squared Error with Decision Tree Regressor: {mse_tree}")
print(f"R-squared (R2) Score with Decision Tree Regressor: {r2_tree}")

# Decision Tree Regressor gave a more accurate prediction
# Recommendation : i will onsider other regression algorithms, feature engineering, hyperparameter tuning, and more to improve model performance.