import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Project directory and data loading
project_dir = os.getenv('PROJECT_DIR', '/path/to/default/dir')
szn_folders = os.listdir(f"{project_dir}/data/pbp")
target_seasons = [x for x in szn_folders if any(yr in x for yr in ['2021', '2022', '2023'])]

# Load data files for target seasons
data_files = [f"{project_dir}/data/pbp/{data_folder}/{os.listdir(f'{project_dir}/data/pbp/{data_folder}')[0]}" 
              for data_folder in target_seasons]

# Concatenate all season data into one DataFrame
dataframes = [pd.read_csv(fn, low_memory=False) for fn in data_files]
df = pd.concat(dataframes, ignore_index=True)

# QB features and grouping
qb_feats = ['season', 'passer_id', 'passer', 'pass', 'complete_pass', 'interception', 'sack', 'yards_gained', 'touchdown']
groupby_feats_qb = ['season', 'passer_id', 'passer']
qb_df = df.loc[:, qb_feats].groupby(groupby_feats_qb, as_index=False).sum()

# Creating previous season for QB
_df_qb = qb_df.copy()
_df_qb['season'] = _df_qb['season'] + 1
new_qb_df = qb_df.merge(_df_qb, on=['season', 'passer_id', 'passer'], suffixes=('', '_prev'), how='left')

# RB features and grouping
rb_feats = ['season', 'rusher_id', 'rusher', 'rush_attempt', 'rushing_yards', 'rush_touchdown']
groupby_feats_rb = ['season', 'rusher_id', 'rusher']
rb_df = df.loc[:, rb_feats].groupby(groupby_feats_rb, as_index=False).sum()

# Creating previous season for RB
_df_rb = rb_df.copy()
_df_rb['season'] = _df_rb['season'] + 1
new_rb_df = rb_df.merge(_df_rb, on=['season', 'rusher_id', 'rusher'], suffixes=('', '_prev'), how='left')

# WR features and grouping
wr_feats = ['season', 'receiver_id', 'receiver', 'pass_attempt', 'receiving_yards', 'yards_after_catch', 'pass_touchdown', 'touchdown']
groupby_feats_wr = ['season', 'receiver_id', 'receiver']
wr_df = df.loc[:, wr_feats].groupby(groupby_feats_wr, as_index=False).sum()

# Creating previous seasons for WR
_df_wr = wr_df.copy()
_df_wr['season'] = _df_wr['season'] + 1
new_wr_df = wr_df.merge(_df_wr, on=['season', 'receiver_id', 'receiver'], suffixes=('', '_prev'), how='left')

# ----- QB Model -----
qb_features = ['pass_prev', 'complete_pass_prev', 'interception_prev', 'sack_prev', 'yards_gained_prev', 'touchdown_prev']
qb_target = 'touchdown'
qb_model_data = new_qb_df.dropna(subset=qb_features + [qb_target])

qb_train_data = qb_model_data[qb_model_data['season'] == 2022]
qb_test_data = qb_model_data[qb_model_data['season'] == 2023]

X_train_qb = qb_train_data[qb_features]
y_train_qb = qb_train_data[qb_target]
X_test_qb = qb_test_data[qb_features]
y_test_qb = qb_test_data[qb_target]

# Polynomial features for QB
poly = PolynomialFeatures(degree=2)
X_train_qb_poly = poly.fit_transform(X_train_qb)
X_test_qb_poly = poly.transform(X_test_qb)

# Linear regression for QB
qb_model = LinearRegression()
qb_model.fit(X_train_qb_poly, y_train_qb)
qb_preds = qb_model.predict(X_test_qb_poly)
qb_test_data = qb_test_data.copy()
qb_test_data.loc[:, 'preds'] = qb_preds

# Evaluating QB model
qb_rmse = mean_squared_error(y_test_qb, qb_preds) ** 0.5
qb_r2 = pearsonr(y_test_qb, qb_preds)[0] ** 2
print(f"\nQB Polynomial Linear Regression Model:\nRMSE: {qb_rmse}\nR²: {qb_r2}")

# ----- RB Model -----
rb_features = ['rush_attempt_prev', 'rushing_yards_prev', 'rush_touchdown_prev']
rb_target = 'rush_touchdown'
rb_model_data = new_rb_df.dropna(subset=rb_features + [rb_target])

rb_train_data = rb_model_data[rb_model_data['season'] == 2022]
rb_test_data = rb_model_data[rb_model_data['season'] == 2023]

X_train_rb = rb_train_data[rb_features]
y_train_rb = rb_train_data[rb_target]
X_test_rb = rb_test_data[rb_features]
y_test_rb = rb_test_data[rb_target]

# Polynomial features for RB
X_train_rb_poly = poly.fit_transform(X_train_rb)
X_test_rb_poly = poly.transform(X_test_rb)

# Linear regression for RB
rb_model = LinearRegression()
rb_model.fit(X_train_rb_poly, y_train_rb)
rb_preds = rb_model.predict(X_test_rb_poly)
rb_test_data = rb_test_data.copy()
rb_test_data.loc[:, 'preds'] = rb_preds

# Evaluating RB model
rb_rmse = mean_squared_error(y_test_rb, rb_preds) ** 0.5
rb_r2 = pearsonr(y_test_rb, rb_preds)[0] ** 2
print(f"\nRB Polynomial Linear Regression Model:\nRMSE: {rb_rmse}\nR²: {rb_r2}")

# ----- WR Model -----
wr_features = ['pass_attempt_prev', 'receiving_yards_prev', 'pass_touchdown_prev']
wr_target = 'pass_touchdown'
wr_model_data = new_wr_df.dropna(subset=wr_features + [wr_target])

wr_train_data = wr_model_data[wr_model_data['season'] == 2022]
wr_test_data = wr_model_data[wr_model_data['season'] == 2023]

X_train_wr = wr_train_data[wr_features]
y_train_wr = wr_train_data[wr_target]
X_test_wr = wr_test_data[wr_features]
y_test_wr = wr_test_data[wr_target]

# Polynomial features for WR
X_train_wr_poly = poly.fit_transform(X_train_wr)
X_test_wr_poly = poly.transform(X_test_wr)

# Linear regression for WR
wr_model = LinearRegression()
wr_model.fit(X_train_wr_poly, y_train_wr)
wr_preds = wr_model.predict(X_test_wr_poly)
wr_test_data = wr_test_data.copy()
wr_test_data.loc[:, 'preds'] = wr_preds

# Evaluating WR model
wr_rmse = mean_squared_error(y_test_wr, wr_preds) ** 0.5
wr_r2 = pearsonr(y_test_wr, wr_preds)[0] ** 2
print(f"\nWR Polynomial Linear Regression Model:\nRMSE: {wr_rmse}\nR²: {wr_r2}")

# ----- Printing QB, RB, WR samples -----
print("\nQB Samples:")
print(qb_test_data[['season', 'passer_id', 'passer', 'touchdown', 'preds']].sample(n=5))

print("\nRB Samples:")
print(rb_test_data[['season', 'rusher_id', 'rusher', 'rush_touchdown', 'preds']].sample(n=5))

print("\nWR Samples:")
print(wr_test_data[['season', 'receiver_id', 'receiver', 'pass_touchdown', 'preds']].sample(n=5))

# Calculating absolute error for QB, RB, WR
qb_test_data.loc[:, 'abs_error'] = abs(qb_test_data['touchdown'] - qb_test_data['preds'])
rb_test_data.loc[:, 'abs_error'] = abs(rb_test_data['rush_touchdown'] - rb_test_data['preds'])
wr_test_data.loc[:, 'abs_error'] = abs(wr_test_data['pass_touchdown'] - wr_test_data['preds'])

# ----- Most Accurate QB Predictions (Touchdowns > 5) -----
qb_test_data_over_5 = qb_test_data[qb_test_data['touchdown'] > 5]
most_accurate_qb_over_5 = qb_test_data_over_5.sort_values(by='abs_error').head(5)
print("\nMost Accurate QB Predictions (Touchdowns > 5):")
print(most_accurate_qb_over_5[['season', 'passer_id', 'passer', 'touchdown', 'preds', 'abs_error']])

# ----- Most Accurate RB Predictions (Touchdowns > 5) -----
rb_test_data_over_5 = rb_test_data[rb_test_data['rush_touchdown'] > 5]
most_accurate_rb_over_5 = rb_test_data_over_5.sort_values(by='abs_error').head(5)
print("\nMost Accurate RB Predictions (Touchdowns > 5):")
print(most_accurate_rb_over_5[['season', 'rusher_id', 'rusher', 'rush_touchdown', 'preds', 'abs_error']])

# ----- Most Accurate WR Predictions (Touchdowns > 5) -----
wr_test_data_over_5 = wr_test_data[wr_test_data['pass_touchdown'] > 5]
most_accurate_wr_over_5 = wr_test_data_over_5.sort_values(by='abs_error').head(5)
print("\nMost Accurate WR Predictions (Touchdowns > 5):")
print(most_accurate_wr_over_5[['season', 'receiver_id', 'receiver', 'pass_touchdown', 'preds', 'abs_error']])