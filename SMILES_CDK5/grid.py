
# 
import pandas as pd

data=pd.read_csv("/home/umang23239/Chemi/MT23239/Input Files/SMILES_CDK5/final_data_label.csv").drop_duplicates(subset='Canonical SMILES',keep='first').dropna(axis=0).reset_index(drop=True)
data

# 
X=data.iloc[:,:-7]
X

# 
y=data['pIC50']
y

# 
data_f=X

# 
for column in data_f.columns:
    if data_f[column].dtype == object:
        data_f[column] = pd.to_numeric(data_f[column], errors='coerce').fillna(0)
    elif pd.api.types.is_numeric_dtype(data_f[column]):
        data_f[column] = pd.to_numeric(data_f[column], errors='coerce').fillna(0)
    else:
        data_f[column] = 0.0
data_f

# %%
from lazypredict.Supervised import LazyClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_f, y,test_size=.2,random_state =123)


# %%


# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)

# %%
X_train_scaled_df=pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_train_scaled_df




# %%


# %%
X_test_scaled_df=pd.DataFrame(X_test_scaled,columns=X_test.columns)
X_test_scaled_df

# # %%
# from lazypredict.Supervised import LazyRegressor

# import numpy as np

# reg = LazyRegressor(verbose=1,ignore_warnings=False, custom_metric=None )
# models,predictions = reg.fit(X_train_scaled_df, X_test_scaled_df, y_train, y_test)
# print(models)

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming you already have the following data:
# X_train_scaled_df, X_test_scaled_df, y_train, y_test

# Define the models and their parameter grids
svr_param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5],
    'kernel': ['linear', 'rbf']
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize models
svr = SVR()
rf = RandomForestRegressor()

# Define the cross-validation scheme
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform Grid Search for SVR
svr_grid = GridSearchCV(svr, svr_param_grid, cv=kf, scoring='neg_mean_squared_error')
svr_grid.fit(X_train_scaled_df, y_train)

# Perform Grid Search for RandomForestRegressor
rf_grid = GridSearchCV(rf, rf_param_grid, cv=kf, scoring='neg_mean_squared_error')
rf_grid.fit(X_train_scaled_df, y_train)

# Get best models
best_svr = svr_grid.best_estimator_
best_rf = rf_grid.best_estimator_

# Get best parameters for each model
print("Best Parameters for SVR:", svr_grid.best_params_)
print("Best Parameters for Random Forest:", rf_grid.best_params_)

# Cross-validation results for SVR
svr_cv_results = cross_val_score(best_svr, X_train_scaled_df, y_train, cv=kf, scoring='neg_mean_squared_error')

# Cross-validation results for RandomForestRegressor
rf_cv_results = cross_val_score(best_rf, X_train_scaled_df, y_train, cv=kf, scoring='neg_mean_squared_error')

# Function to calculate additional metrics
def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return mse, rmse, r2, adj_r2, mae

# Train metrics for SVR
print("SVR - Cross-Validation Results (MSE):", -svr_cv_results.mean())
mse, rmse, r2, adj_r2, mae = calculate_metrics(best_svr, X_train_scaled_df, y_train)
print(f"SVR - Train Metrics:\nMSE: {mse}\nRMSE: {rmse}\nR²: {r2}\nAdjusted R²: {adj_r2}\nMAE: {mae}")

# Train metrics for RandomForestRegressor
print("Random Forest - Cross-Validation Results (MSE):", -rf_cv_results.mean())
mse, rmse, r2, adj_r2, mae = calculate_metrics(best_rf, X_train_scaled_df, y_train)
print(f"Random Forest - Train Metrics:\nMSE: {mse}\nRMSE: {rmse}\nR²: {r2}\nAdjusted R²: {adj_r2}\nMAE: {mae}")

# Predict and evaluate on Test Data (for both models)

# SVR Test Metrics
svr_test_metrics = calculate_metrics(best_svr, X_test_scaled_df, y_test)
print(f"SVR - Test Metrics:\nMSE: {svr_test_metrics[0]}\nRMSE: {svr_test_metrics[1]}\nR²: {svr_test_metrics[2]}\nAdjusted R²: {svr_test_metrics[3]}\nMAE: {svr_test_metrics[4]}")

# Random Forest Test Metrics
rf_test_metrics = calculate_metrics(best_rf, X_test_scaled_df, y_test)
print(f"Random Forest - Test Metrics:\nMSE: {rf_test_metrics[0]}\nRMSE: {rf_test_metrics[1]}\nR²: {rf_test_metrics[2]}\nAdjusted R²: {rf_test_metrics[3]}\nMAE: {rf_test_metrics[4]}")
