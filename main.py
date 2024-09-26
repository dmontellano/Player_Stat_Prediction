import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr


class NFLStatPrediction:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.poly = PolynomialFeatures(degree=2)

    def load_data(self):
        """Load season data and concatenate into a single DataFrame."""
        szn_folders = os.listdir(f"{self.project_dir}/data/pbp")
        target_seasons = [x for x in szn_folders if any(yr in x for yr in ['2021', '2022', '2023'])]
        
        data_files = [f"{self.project_dir}/data/pbp/{data_folder}/{os.listdir(f'{self.project_dir}/data/pbp/{data_folder}')[0]}" 
                      for data_folder in target_seasons]

        dataframes = [pd.read_csv(fn, low_memory=False) for fn in data_files]
        df = pd.concat(dataframes, ignore_index=True)
        return df

    def create_previous_season_data(self, df, feats, groupby_feats):
        """Group and create previous season's stats for a specific position."""
        grouped_df = df.loc[:, feats].groupby(groupby_feats, as_index=False).sum()

        _df_prev = grouped_df.copy()
        _df_prev['season'] = _df_prev['season'] + 1
        new_df = grouped_df.merge(_df_prev, on=groupby_feats, suffixes=('', '_prev'), how='left')
        return new_df

    def train_model(self, model_data, features, target, train_season, test_season):
        """Train and evaluate a polynomial linear regression model."""
        # Split data into train and test sets
        train_data = model_data[model_data['season'] == train_season]
        test_data = model_data[model_data['season'] == test_season]

        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]

        # Polynomial transformation
        X_train_poly = self.poly.fit_transform(X_train)
        X_test_poly = self.poly.transform(X_test)

        # Train the model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        preds = model.predict(X_test_poly)

        # Evaluation
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = pearsonr(y_test, preds)[0] ** 2

        test_data = test_data.copy()
        test_data['preds'] = preds
        return model, test_data, rmse, r2

    def evaluate_model(self, test_data, target, threshold=5):
        """Evaluate model by calculating absolute error and showing most accurate predictions."""
        test_data['abs_error'] = abs(test_data[target] - test_data['preds'])
        test_data_over_thresh = test_data[test_data[target] > threshold]
        most_accurate = test_data_over_thresh.sort_values(by='abs_error').head(5)
        return most_accurate

    def run_models(self):
        """Run models for QB, RB, and WR and print results."""
        df = self.load_data()

        # QB features and model
        qb_feats = ['season', 'passer_id', 'passer', 'pass', 'complete_pass', 'interception', 'sack', 'yards_gained', 'touchdown']
        qb_groupby_feats = ['season', 'passer_id', 'passer']
        qb_df = self.create_previous_season_data(df, qb_feats, qb_groupby_feats)

        qb_features = ['pass_prev', 'complete_pass_prev', 'interception_prev', 'sack_prev', 'yards_gained_prev', 'touchdown_prev']
        qb_target = 'touchdown'
        qb_model, qb_test_data, qb_rmse, qb_r2 = self.train_model(qb_df.dropna(subset=qb_features + [qb_target]), qb_features, qb_target, 2022, 2023)
        print(f"QB Polynomial Linear Regression Model:\nRMSE: {qb_rmse}\nR²: {qb_r2}")
        print(self.evaluate_model(qb_test_data, qb_target))

        # RB features and model
        rb_feats = ['season', 'rusher_id', 'rusher', 'rush_attempt', 'rushing_yards', 'rush_touchdown']
        rb_groupby_feats = ['season', 'rusher_id', 'rusher']
        rb_df = self.create_previous_season_data(df, rb_feats, rb_groupby_feats)

        rb_features = ['rush_attempt_prev', 'rushing_yards_prev', 'rush_touchdown_prev']
        rb_target = 'rush_touchdown'
        rb_model, rb_test_data, rb_rmse, rb_r2 = self.train_model(rb_df.dropna(subset=rb_features + [rb_target]), rb_features, rb_target, 2022, 2023)
        print(f"RB Polynomial Linear Regression Model:\nRMSE: {rb_rmse}\nR²: {rb_r2}")
        print(self.evaluate_model(rb_test_data, rb_target))

        # WR features and model
        wr_feats = ['season', 'receiver_id', 'receiver', 'pass_attempt', 'receiving_yards', 'yards_after_catch', 'pass_touchdown', 'touchdown']
        wr_groupby_feats = ['season', 'receiver_id', 'receiver']
        wr_df = self.create_previous_season_data(df, wr_feats, wr_groupby_feats)

        wr_features = ['pass_attempt_prev', 'receiving_yards_prev', 'pass_touchdown_prev']
        wr_target = 'pass_touchdown'
        wr_model, wr_test_data, wr_rmse, wr_r2 = self.train_model(wr_df.dropna(subset=wr_features + [wr_target]), wr_features, wr_target, 2022, 2023)
        print(f"WR Polynomial Linear Regression Model:\nRMSE: {wr_rmse}\nR²: {wr_r2}")
        print(self.evaluate_model(wr_test_data, wr_target))

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

project_dir = os.getenv('PROJECT_DIR', '/path/to/default/dir')
nfl_stat_prediction = NFLStatPrediction(project_dir)
nfl_stat_prediction.run_models()