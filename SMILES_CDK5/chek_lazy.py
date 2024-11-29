# # Importing the pandas library for data manipulation and analysis
import pandas as pd
# Reading a CSV file into a DataFrame, removing duplicates, handling missing values, and resetting the index
data=pd.read_csv("/home/umang23239/Chemi/MT23239/Input Files/SMILES_CDK5/final_data_label.csv").drop_duplicates(subset='Canonical SMILES',keep='first').dropna(axis=0).reset_index(drop=True)
data # Displaying the processed DataFrame

# 
X=data.iloc[:,:-7] # Selecting all columns except the last 7 columns from the DataFrame
X

# 
y=data['pIC50'] # Extracting the 'pIC50' column as the target variable
y

# 
data_f=X

# Convert all columns in the DataFrame to numeric, replacing invalid or unsupported values with 0.

for column in data_f.columns:
    if data_f[column].dtype == object:
        data_f[column] = pd.to_numeric(data_f[column], errors='coerce').fillna(0)
    elif pd.api.types.is_numeric_dtype(data_f[column]):
        data_f[column] = pd.to_numeric(data_f[column], errors='coerce').fillna(0)
    else:
        data_f[column] = 0.0
data_f

# Split the dataset into training and testing sets and import LazyClassifier for automated model benchmarking.

from lazypredict.Supervised import LazyClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_f, y,test_size=.2,random_state =123)




#Scale the training and testing feature sets using StandardScaler for normalization.

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)

# Create a DataFrame for the scaled training data, retaining the original column names.
X_train_scaled_df=pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_train_scaled_df


#Create a DataFrame for the scaled testing data, retaining the original column names.
X_test_scaled_df=pd.DataFrame(X_test_scaled,columns=X_test.columns)
X_test_scaled_df

from lazypredict.Supervised import LazyRegressor

import numpy as np

#Initialize and fit a LazyRegressor to evaluate multiple regression models, generating performance metrics and predictions.
reg = LazyRegressor(verbose=1,ignore_warnings=False, custom_metric=None )
models,predictions = reg.fit(X_train_scaled_df, X_test_scaled_df, y_train, y_test)
print(models)


