##########################################
## Script by Bradley T. Martin
##########################################

import argparse
import os
import sys
from pathlib import Path
import datetime
import requests
import pickle
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


from pybaseball import (
    statcast,
    statcast_batter_expected_stats,
    statcast_pitcher_expected_stats,
    statcast_batter,
    statcast_pitcher,
    batting_stats,
    batting_stats_range,
    pitching_stats,
    pitching_stats_range,
    playerid_lookup,
    cache,
    playerid_reverse_lookup,
)

# For faster data retrieval when running multiple times.
cache.disable()

from feature_engine.selection import (
    SmartCorrelatedSelection,
    RecursiveFeatureElimination,
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.selection import RecursiveFeatureElimination
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import cross_val_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.base import BaseEstimator, RegressorMixin
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.layers import LSTM, Dense, Dropout


class LSTMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_features,
        window_size,
    ):
        self.n_features = n_features
        self.window_size = window_size

    # def fit(self, X, y):
    #     self.n_features = X.shape[2]
    #     self.model_ = self._create_model()
    #     self.model_.fit(X, y)
    #     return self

    # def predict(self, X):
    #     return self.model_.predict(X)

    def _create_model(
        self,
        neurons=100,
        dropout=0.2,
        recurrent_dropout=0.2,
        optimizer="adam",
        learning_rate=0.1,
    ):
        model = Sequential()
        model.add(
            LSTM(
                neurons,
                return_sequences=True,
                input_shape=(self.window_size, self.n_features),
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )
        model.add(
            LSTM(neurons, dropout=dropout, recurrent_dropout=recurrent_dropout)
        )
        model.add(Dense(1))

        if optimizer.lower() == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == "sgd":
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer.lower() == "rmsprop":
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(
                "Invalid optimizer provided; optimizer must be either 'adam', 'sgd', or 'rmsprop'"
            )

        model.compile(loss="mean_squared_error", optimizer=optimizer)
        return model


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        X_transformed = self.transformer.transform(X)
        return pd.DataFrame(X_transformed, index=X.index, columns=X.columns)


class DataPreparationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, threshold=2):
        self.features = features
        self.threshold = threshold

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            self.n_features_ = len(X.columns)
        else:
            self.feature_names_in_ = f"x{list(range(len(X)))}"
            self.n_features_ = X.shape[1]
        return self

    def transform(self, X, target=False):
        X = X.copy()
        self.features = [x for x in self.features if x in X.columns]

        # Convert the relevant columns to numeric data type
        for feature in self.features:
            X[feature] = pd.to_numeric(X[feature], errors="coerce")

        if target:
            print(X)
        # Create a new column for each feature that represents the percentage change in the feature
        for feature in self.features:
            # Calculate percentage change
            X[feature + "_change"] = X.groupby("IDfg")[feature].pct_change()

            # Calculate Z-scores
            X[feature + "_zscore"] = X.groupby("IDfg")[
                feature + "_change"
            ].transform(lambda x: (x - x.mean()) / x.std())

        if target:
            print(X)

        # Summarize the targets into one predictive value by adding up the individual targets
        X["target"] = X[
            [feature + "_zscore" for feature in self.features]
        ].sum(axis=1)

        if target:
            print(X)

        # Drop the first row for each player, which will have NA values for the change and target variables
        X = (
            X.groupby("IDfg")
            .apply(lambda x: x.iloc[1:])
            .reset_index(drop=True)
        )

        if target:
            print(X)

        X["IDfg_zscore"] = X["IDfg"]
        y = X["target"]
        X_data = X[[feature + "_change" for feature in self.features]]
        X_data.columns = X_data.columns.str.replace("_zscore", "")

        if target:
            print(X_data)

        return X_data, y

    def get_target(self, X):
        # Ensure the target is calculated
        if "target" not in X.columns:
            y = self.transform(X, target=True)
        return y


class RegressionCandidate:
    def __init__(
        self,
        player_data,
        player_type,
        window_size=2,
        param_grid=None,
        features=None,
        lstm=True,
        verbose=True,
    ):
        self.player_data = player_data.copy()
        self.player_type = player_type
        self.player_model = None
        self.window_size = window_size
        self.features = features
        self.lstm = lstm
        self.verbose = verbose

        self.info_cols = [
            "Age",
            "G",
            "AB",
            "PA",
            "IP",
        ]

        if param_grid is None:
            self.do_gridsearch = False
            param_grid = {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "colsample_bytree": [0.7, 0.9, 1],
            }
        else:
            self.do_gridsearch = True
        self.param_grid = param_grid
        self.params = {
            "colsample_bytree": 0.7,
            "learning_rate": 0.2,
            "max_depth": 3,
            "n_estimators": 150,
        }

    def calculate_target(self, sequence, features):
        # Calculate the z-scores for each feature
        for feature in features:
            if sequence[feature].std() != 0:
                sequence[feature] = zscore(sequence[feature])
            else:
                sequence[feature] = 0  # or any other default value

        # Define features where a higher value is worse for performance
        negative_impact_features_hitters = ["O-Swing%", "SwStr%"]
        negative_impact_features_pitchers = ["AVG", "WHIP", "BABIP", "Barrel%"]

        # Check if these features exist in your data and adjust their sign
        for feature in negative_impact_features_hitters:
            if feature in sequence.columns:
                sequence[feature] = -sequence[feature]

        for feature in negative_impact_features_pitchers:
            if feature in sequence.columns:
                sequence[feature] = -sequence[feature]

        # Sum the z-scores to create the 'target' value
        target = sequence[features].sum(axis=1).mean()

        return target

    def get_train_test_lstm(
        self, data, features, window_size=2, training=False
    ):
        data = data.copy()
        features = [
            x
            for x in features
            if x != "IDfg" and x != "Name" and x != "Season" and x != "Month"
        ]

        # Convert the relevant columns to numeric data type
        for feature in features:
            data[feature] = pd.to_numeric(data[feature], errors="coerce")

        # Impute missing data.
        self.imputer = DataFrameTransformer(SimpleImputer(strategy="median"))
        data[features] = self.imputer.fit_transform(data[features])

        # Normalize features
        self.scaler = DataFrameTransformer(StandardScaler())
        data[features] = self.scaler.fit_transform(data[features])

        # Sort by IDfg, Season, and Month before doing the rolling window calculations
        data = data.sort_values(["IDfg", "Season", "Month"])

        # Create sequences for each player
        sequences = []
        targets = []
        for player in data["IDfg"].unique():
            player_data = data[data["IDfg"] == player]

            if len(player_data) < window_size:
                continue

            for i in range(len(player_data) - window_size + 1):
                sequence = player_data.iloc[i : i + window_size][features]
                target = self.calculate_target(sequence, features)
                sequences.append(sequence.values)
                targets.append(target)

        def print_nan_cols(df):
            nan_cols = df.columns[df.isna().any()].tolist()
            print(nan_cols)

        X = np.array(sequences)
        y = np.array(targets)

        if training:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        else:
            return X, y

    def get_train_test(self, data, features, training=False, window_size=2):
        data = data.copy()
        features = [x for x in features if x != "IDfg" and x != "Name"]

        # Convert the relevant columns to numeric data type
        for feature in features:
            data[feature] = pd.to_numeric(data[feature], errors="coerce")

        # Normalize features
        scaler = StandardScaler()
        data[features] = scaler.fit_transform(data[features])

        # Sort by IDfg, Season, and Month before doing the rolling window calculations
        data = data.sort_values(["IDfg", "Season", "Month"])

        # Calculate the rolling mean of each feature and the change from the rolling mean
        for feature in features:
            data[feature + "_mean"] = data.groupby("IDfg")[feature].transform(
                lambda x: x.rolling(window_size).mean()
            )
            data[feature + "_change"] = data[feature] - data[feature + "_mean"]

        # Create a target variable that represents the average change for each feature
        data["target"] = data[
            [feature + "_change" for feature in features]
        ].mean(axis=1)

        # Drop rows with NaN values in the target column
        data = data.dropna(subset=["target"])

        # Drop the first window_size rows for each player, which will have NA values for the change and target variables
        data = (
            data.groupby("IDfg")
            .apply(
                lambda x: x
                if x.shape[0] <= window_size
                else x.iloc[window_size:]
            )
            .reset_index(drop=True)
        )

        X = data[features]
        y = data["target"]
        idfg = data[["IDfg", "Name"]]
        X = X.drop(
            ["IDfg", "Name", "Season", "Month"], axis=1, errors="ignore"
        )

        if training:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        else:
            return X, y, idfg

    def prepare_data(self, data, features, training=False, lstm=True):
        info_data = data.loc[:, data.columns.isin(self.info_cols)]
        transformed_data = data.loc[:, ~data.columns.isin(self.info_cols)]
        transformed_data = transformed_data.loc[:, features]
        transformed_data = transformed_data.dropna(axis=1, how="all")
        features = list(transformed_data.columns)

        func = self.get_train_test_lstm if lstm else self.get_train_test
        if training:
            X_train, X_test, y_train, y_test = func(
                transformed_data,
                features,
                window_size=self.window_size,
                training=training,
            )
            return X_train, X_test, y_train, y_test, info_data
        else:
            X, y, idfg = func(
                transformed_data,
                features,
                window_size=self.window_size,
                training=training,
            )

            return X, y, idfg

    def reshape_data(self, df, sequence_length):
        sequences = []
        for player in df["Name"].unique():
            player_data = df[df["Name"] == player]
            for i in range(len(player_data) - sequence_length):
                sequences.append(
                    player_data[i : i + sequence_length + 1]
                    .drop(columns=["IDfg", "Name"])
                    .values
                )
        return np.array(sequences)

    def train_models(self):
        func = (
            self._prepare_and_run_lstm_model
            if self.lstm
            else self._prepare_and_run_model
        )

        if Path("results/train_test_data.pkl").is_file():
            with open("results/train_test_data.pkl", "rb") as fin:
                train_data = pickle.load(fin)
                (
                    X_train_player,
                    X_test_player,
                    y_train_player,
                    y_test_player,
                ) = train_data
        else:
            (
                X_train_player,
                X_test_player,
                y_train_player,
                y_test_player,
                self._player_info_,
            ) = self.prepare_data(
                self.player_data, self.features, training=True, lstm=self.lstm
            )
            with open("results/train_test_data.pkl", "wb") as fout:
                pickle.dump(
                    (
                        X_train_player,
                        X_test_player,
                        y_train_player,
                        y_test_player,
                    ),
                    fout,
                )

        print(f"Running {self.player_type} model...")

        (
            self.player_model,
            self.player_pipeline,
        ) = func(
            X_train_player,
            y_train_player,
            self.param_grid,
            self.player_type,
        )

        print(f"Cross-validating holdout {self.player_type} test dataset...")
        self.xval(X_test_player, y_test_player, cv=5)

    @staticmethod
    def model_build_fn(compile_kwargs, **kwargs):
        dropout = kwargs.get("dropout", 0.2)
        window_size = kwargs.get("window_size", 2)
        n_features = kwargs.get("n_features", 12)
        recurrent_dropout = kwargs.get("recurrent_dropout", 0.2)
        neurons = kwargs.get("neurons", 100)

        model = Sequential()
        model.add(
            LSTM(
                neurons,
                return_sequences=True,
                input_shape=(window_size, n_features),
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )
        model.add(
            LSTM(neurons, dropout=dropout, recurrent_dropout=recurrent_dropout)
        )
        model.add(Dense(1))
        model.compile(**compile_kwargs)
        return model

    def _prepare_and_run_lstm_model(
        self,
        X_train,
        y_train,
        param_grid,
        player_type,
    ):
        model_callable = RegressionCandidate.model_build_fn
        if self.do_gridsearch:
            additional_params = {
                "model__window_size": X_train.shape[1],
                "model__n_features": X_train.shape[2],
            }
        else:
            self.params["model__window_size"] = X_train.shape[1]
            self.params["model__n_features"] = X_train.shape[2]

        if not self.do_gridsearch:
            if self.params is None:
                raise TypeError(
                    "params and param_grid cannot both be NoneType"
                )
            if self.verbose:
                print(f"Training {player_type} model...")

            pipeline = make_pipeline(
                KerasRegressor(model_callable, **self.params),
            )
        else:
            if self.verbose:
                print(f"Doing {player_type} grid search...")
            pipeline = make_pipeline(
                GridSearchCV(
                    KerasRegressor(
                        model=model_callable,
                        optimizer=Adam,
                        loss="mean_squared_error",
                        verbose=0,
                        **additional_params,
                    ),
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    verbose=1,
                ),
            )

        self.n_features = X_train.shape[1]

        pipeline.fit(X_train, y_train)

        if not self.do_gridsearch:
            model = pipeline[-1]
        else:
            grid_search = pipeline[-1]
            model = grid_search.best_estimator_
            if self.verbose:
                print(
                    f"Best {player_type} train score: {grid_search.best_score_}"
                )
                print(f"Best {player_type} params: {grid_search.best_params_}")

            self.plot_grid_search_results(grid_search, player_type)
        self.plot_feature_importances(model, X_train, player_type)

        if self.verbose:
            print(f"DONE TRAINING {player_type.upper()} MODEL!\n\n")
        return model, pipeline

    # def create_model(
    #     self, neurons=100, optimizer="adam", dropout=0.2, recurrent_dropout=0.2
    # ):
    #     model = Sequential()
    #     model.add(
    #         LSTM(
    #             neurons,
    #             return_sequences=True,
    #             input_shape=(self.n_features, 1),
    #             dropout=dropout,
    #             recurrent_dropout=recurrent_dropout,
    #         )
    #     )
    #     model.add(
    #         LSTM(neurons, dropout=dropout, recurrent_dropout=recurrent_dropout)
    #     )
    #     model.add(Dense(1))
    #     model.compile(loss="mean_squared_error", optimizer=optimizer)
    #     return model

    def _prepare_and_run_model(
        self,
        X_train,
        y_train,
        param_grid,
        player_type,
    ):
        if not self.do_gridsearch:
            if self.params is None:
                raise TypeError(
                    "params and param_grid cannot both be NoneType"
                )
            if self.verbose:
                print(f"Training {player_type} model...")
            pipeline = make_pipeline(
                DataFrameTransformer(SimpleImputer(strategy="median")),
                XGBRegressor(**self.params),
            )
        else:
            if self.verbose:
                print(f"Doing {player_type} grid search...")
            pipeline = make_pipeline(
                DataFrameTransformer(SimpleImputer(strategy="median")),
                GridSearchCV(
                    XGBRegressor(),
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                ),
            )

        pipeline.fit(X_train, y_train)

        if not self.do_gridsearch:
            model = pipeline[-1]
        else:
            grid_search = pipeline[-1]
            model = grid_search.best_estimator_
            if self.verbose:
                print(
                    f"Best {player_type} train score: {grid_search.best_score_}"
                )
                print(f"Best {player_type} params: {grid_search.best_params_}")

            self.plot_grid_search_results(grid_search, player_type)
        self.plot_feature_importances(model, X_train, player_type)

        if self.verbose:
            print(f"DONE TRAINING {player_type.upper()} MODEL!\n\n")
        return model, pipeline

    def predict_regression(self, model, X):
        X = X.copy()
        if "IDfg" in X.columns:
            X = X.drop(["IDfg"], axis=1)
        if "target" in X.columns:
            X = X.drop(["target"], axis=1)
        return model.predict(X)

    def xval(self, X_test, y_test, cv=5):
        scores = cross_val_score(
            self.player_model,
            X_test,
            y_test,
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        self.scores = scores
        print(
            f"{self.player_type.capitalize()} cross-validation scores: {scores}"
        )
        print(
            f"{self.player_type.capitalize()} average cross-validation score: {scores.mean()}"
        )

    def plot_feature_importances(self, model, X, player_type):
        print("Plotting...")

        outfile = f"results/feature_importances_{player_type}.png"

        # Get feature importances
        importances = model.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(10, 6))

        # Create plot title
        plt.title("Feature Importance")

        # Add bars
        plt.bar(range(X.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X.shape[1]), names, rotation=90)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")

    def plot_grid_search_results(self, grid_search, player_type):
        results = pd.DataFrame(grid_search.cv_results_)
        params = list(grid_search.param_grid.keys())

        outfile = f"results/gridsearch_results_{player_type}.png"

        # Prepare a DataFrame in a way suitable for sns.PairGrid
        results_melted = results.melt(
            id_vars=["rank_test_score"],
            value_vars=[f"param_{param}" for param in params],
            var_name="Parameters",
            value_name="Values",
        )

        # Initialize a PairGrid
        grid = sns.PairGrid(results_melted, diag_sharey=False, corner=True)

        # Map a histogram to the diagonal
        grid.map_diag(sns.histplot)

        # Map a boxplot to the off-diagonal subplots
        grid.map_lower(sns.boxplot)

        # Save the figure
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")

    def marcel_projection(self, player_data):
        player_type = player_data.pop("player_type")
        player_id = player_data.pop("player_id")
        player_age = player_data.pop("age")
        age_factor = self.get_age_factor(player_age)

        # We'll need to retrieve the player's data for the past three seasons
        past_data = self.get_past_data(player_type, player_id)

        # Calculate weighted averages for each stat, with more recent seasons weighted more heavily
        # We also apply regression and the age factor
        for stat in past_data.columns:
            player_data[stat] = (
                (
                    5 * past_data[stat].iloc[0]
                    + 4 * past_data[stat].iloc[1]
                    + 3 * past_data[stat].iloc[2]
                )
                / 12
                * age_factor
            )

        return self.predict_regression(player_data)

    def get_age_factor(self, age):
        if age < 27:
            return 1.1
        elif 27 <= age <= 33:
            return 1
        else:
            return 0.9

    def get_past_data(self, player_type, player_id):
        data = self.player_data
        return (
            data[data["player_id"] == player_id]
            .sort_values(by="season", ascending=False)
            .head(3)
        )


def scrape_periods(interval=30, func=pitching_stats):
    # Create an empty list to store the DataFrames
    dataframes = []

    months = list(range(4, 10))
    for season in range(2015, 2024):
        for month in months:
            try:
                data = func(season, split_seasons=False, month=month)
                data["Season"] = season
                data["Month"] = month
            except ValueError:
                continue

            # Add the data to the list of DataFrames
            dataframes.append(data)

    # Concatenate all the dataframes in the list
    all_data = pd.concat(dataframes, ignore_index=True)

    return all_data


# def make_features(df, df2):
#     df["BABIP_AVG_DIFF"] = df["BABIP"] - df["AVG"]
#     df.drop(["AVG", "BABIP"], axis=1, inplace=True)

#     df2["xFIP_ERA_DIFF"] = df2["xFIP"] - df2["ERA"]
#     df2.drop(["FIP", "ERA", "xFIP"], axis=1, inplace=True)

#     return df, df2


# def main():
#     if Path("results/stats_pickled/stats_pitcher_fg.pkl").is_file():
#         with open("results/stats_pickled/stats_pitcher_fg.pkl", "rb") as fin:
#             dfpitch = pickle.load(fin)
#     else:
#         dfpitch = scrape_periods(interval=30, func=pitching_stats)
#         with open("results/stats_pickled/stats_pitcher_fg.pkl", "wb") as fout:
#             pickle.dump(dfpitch, fout)

#     if Path("results/stats_pickled/stats_hitter_fg.pkl").is_file():
#         with open("results/stats_pickled/stats_hitter_fg.pkl", "rb") as fin:
#             dfhit = pickle.load(fin)
#     else:
#         dfhit = scrape_periods(interval=30, func=batting_stats)
#         with open("results/stats_pickled/stats_hitter_fg.pkl", "wb") as fout:
#             pickle.dump(dfhit, fout)

#     dfpitch["player_type"] = "pitcher"
#     dfhit["player_type"] = "hitter"

#     dfpitch["Season"] = dfpitch["Season"].astype(int)
#     dfhit["Season"] = dfhit["Season"].astype(int)

#     dfhit, dfpitch = make_features(dfhit, dfpitch)

#     dfpitch_23 = dfpitch[dfpitch["Season"] == 2023]
#     dfhit_23 = dfhit[dfhit["Season"] == 2023]
#     dfhit_23 = dfhit_23[dfhit_23["Month"].isin([5, 6])]
#     dfpitch_23 = dfpitch_23[dfpitch_23["Month"].isin([5, 6])]

#     dfhit_23 = dfhit_23.sort_values(by="IDfg", ascending=True)
#     dfpitch_23 = dfpitch_23.sort_values(by="IDfg", ascending=True)

#     dfpitch = dfpitch[dfpitch["Season"] != 2023]
#     dfhit = dfhit[dfhit["Season"] != 2023]

#     dfhit = dfhit.sort_values(by="IDfg", ascending=True)

#     if Path("results/rc.pkl").is_file():
#         with open("results/rc.pkl", "rb") as fin:
#             rc = pickle.load(fin)
#     else:
#         # Initialize the RegressionCandidate object
#         rc = RegressionCandidate(dfhit, dfpitch, run="all")
#         # Train the hitter model
#         rc.train_models()
#         with open("results/rc.pkl", "wb") as fout:
#             pickle.dump(rc, fout)

#     dfpitch_23, _, dfp_id23 = rc.prepare_data(dfpitch_23, rc.pitcher_features)
#     dfhit_23, _, dfh_id23 = rc.prepare_data(dfhit_23, rc.hitter_features)

#     transformers = rc.hitter_pipeline[:-1]

#     for t in transformers:
#         dfhit_23 = t.transform(dfhit_23)

#     transformers = rc.pitcher_pipeline[:-1]

#     for t in transformers:
#         dfpitch_23 = t.transform(dfpitch_23)

#     # Make the predictions.
#     pitch_pred = rc.predict_regression(rc.pitcher_model, dfpitch_23)
#     hit_pred = rc.predict_regression(rc.hitter_model, dfhit_23)

#     dfh_id23["Prediction"] = hit_pred
#     dfp_id23["Prediction"] = pitch_pred
#     outpitch = dfp_id23.copy()
#     outhit = dfh_id23.copy()

#     outpitch = outpitch.sort_values(by="Prediction", ascending=False)
#     outhit = outhit.sort_values(by="Prediction", ascending=False)
#     outpitch.to_csv("results/pitcher_pred.csv", header=True, index=False)
#     outhit.to_csv("results/hitter_pred.csv", header=True, index=False)


def make_features(
    df, feature_cols_to_drop, diff_feature1, diff_feature2, new_feature_name
):
    df[new_feature_name] = df[diff_feature1] - df[diff_feature2]
    df.drop(feature_cols_to_drop, axis=1, inplace=True)
    return df


def load_or_scrape(filename, scrape_func):
    if Path(filename).is_file():
        with open(filename, "rb") as fin:
            df = pickle.load(fin)
    else:
        df = scrape_periods(interval=30, func=scrape_func)
        with open(filename, "wb") as fout:
            pickle.dump(df, fout)
    return df


def process_data(df, season, months, sort_by, player_type):
    df["player_type"] = player_type
    df["Season"] = df["Season"].astype(int)
    df_current_season = df[df["Season"] == season]
    df_current_season = df_current_season[
        df_current_season["Month"].isin(months)
    ]
    df_current_season = df_current_season.sort_values(
        by=sort_by, ascending=True
    )
    df = df[df["Season"] != season]
    df = df.sort_values(by=sort_by, ascending=True)
    return df, df_current_season


def load_or_train_model(
    filename, df, player_type, features, window_size, param_grid
):
    if Path(filename).is_file():
        with open(filename, "rb") as fin:
            rc = pickle.load(fin)
    else:
        rc = RegressionCandidate(
            df,
            player_type,
            window_size=window_size,
            param_grid=param_grid,
            features=features,
            lstm=True,
            verbose=True,
        )
        rc.train_models()
        with open(filename, "wb") as fout:
            pickle.dump(rc, fout)
    return rc


def preprocess_data(df, pipeline, rc=None, lstm=False):
    if lstm and rc is None:
        raise TypeError("rc cannot be NoneType if lstm is True")

    if lstm:
        df = rc.imputer.transform(df)
        df = rc.scaler.transform(df)
    else:
        transformers = pipeline[:-1]
        for t in transformers:
            df = t.transform(df)
        return df


def run_pipeline(
    player_type, window_size, features, param_grid=None, lstm=True
):
    if player_type == "pitcher":
        stats_filename = "results/stats_pickled/stats_pitcher_fg.pkl"
        scrape_func = pitching_stats
        # feature_cols_to_drop = ["FIP", "ERA", "xFIP"]
        # diff_feature1 = "xFIP"
        # diff_feature2 = "ERA"
        # new_feature_name = "xFIP_ERA_DIFF"
        rc_filename = "results/rc_pitcher.pkl"
        df = load_or_scrape(stats_filename, scrape_func)
        # df = make_features(
        #     df,
        #     feature_cols_to_drop,
        #     diff_feature1,
        #     diff_feature2,
        #     new_feature_name,
        # )
        df, df_current_season = process_data(
            df, 2023, [5, 6], "IDfg", player_type
        )
        rc = load_or_train_model(
            rc_filename, df, player_type, features, window_size, param_grid
        )
        features = rc.features
        pipeline = rc.player_pipeline
        model = rc.player_model
        output_filename = "results/pitcher_pred.csv"
    else:
        stats_filename = "results/stats_pickled/stats_hitter_fg.pkl"
        scrape_func = batting_stats
        # feature_cols_to_drop = ["AVG", "BABIP"]
        # diff_feature1 = "BABIP"
        # diff_feature2 = "AVG"
        # new_feature_name = "BABIP_AVG_DIFF"
        rc_filename = "results/rc_hitter.pkl"
        df = load_or_scrape(stats_filename, scrape_func)
        cols = list(df.columns)
        cols.sort()
        print(
            [
                x
                for x in cols
                if not x.endswith("+")
                and not x.endswith("(pi)")
                and not x.endswith("(sc)")
                and not x.endswith("-")
                and not x.startswith("+")
            ]
        )
        sys.exit()
        # df = make_features(
        #     df,
        #     feature_cols_to_drop,
        #     diff_feature1,
        #     diff_feature2,
        #     new_feature_name,
        # )
        df, df_current_season = process_data(
            df, 2023, [5, 6], "IDfg", player_type
        )

        rc = load_or_train_model(
            rc_filename, df, player_type, features, window_size, param_grid
        )
        features = rc.features

        pipeline = rc.player_pipeline
        model = rc.player_model
        output_filename = "results/hitter_pred.csv"

    df_current_season, _, df_ids = rc.prepare_data(df_current_season, features)
    df_current_season = preprocess_data(
        df_current_season, pipeline, rc=rc, lstm=lstm
    )
    predictions = rc.predict_regression(model, df_current_season)
    df_ids["Prediction"] = predictions
    df_ids = df_ids.sort_values(by="Prediction", ascending=False)
    df_ids.to_csv(output_filename, header=True, index=False)

    plot_data(df_ids, player_type)

    # Assuming data is your DataFrame


def plot_data(data, player_type):
    plt.figure(figsize=(20, 30))
    sns.barplot(x="Prediction", y="Name", data=data, orient="h")
    plt.axvline(0, color="red")
    plt.title(f"{player_type.capitalize()} Regression Predictions")
    plt.xlabel("Prediction (Std Dev)")
    plt.ylabel("Name")
    plt.savefig(f"results/pred_plot_{player_type}")


def main():
    hitter_features = [
        "IDfg",
        "Name",
        "Season",
        "Month",
        "AVG",
        "BABIP",
        "BB/K",
        "Barrel%",
        "LD%",
        "O-Swing%",
        "SwStr%",
        "wOBA",
        "wRC+",
    ]

    pitcher_features = [
        "IDfg",
        "Name",
        "Season",
        "Month",
        "AVG",
        "BABIP",
        "Barrel%",
        "E-F",
        "FBv",
        "GB/FB",
        "K-BB%",
        "O-Swing%",
        "SwStr%",
        "WAR",
        "WHIP",
    ]

    # param_grid = {
    #     "n_estimators": [50, 100, 150, 200],
    #     "learning_rate": [0.01, 0.1, 0.2],
    #     "max_depth": [2, 3, 5, 7],
    #     "colsample_bytree": [0.5, 0.55, 0.6, 0.65],
    #     "reg_alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    #     "reg_lambda": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    # }

    param_grid = {
        "model__neurons": [32, 64, 128],
        "model__dropout": [0.0, 0.1, 0.2],
        "optimizer__learning_rate": [0.1, 0.01, 0.001],
        # "lstmregressor__optimizer": ["adam", "sgd", "rmsprop"],
    }

    run_pipeline("hitter", 2, hitter_features, param_grid, lstm=True)
    # run_pipeline("pitcher", 2, pitcher_features, param_grid, lstm=True)


# # Perform a Marcel Projection for a specific hitter
# hitter_projection = rc.marcel_projection({'player_id': 1, 'age': 25, 'player_type': 'hitter'})

# # Perform a Marcel Projection for a specific pitcher
# pitcher_projection = rc.marcel_projection({'player_id': 2, 'age': 30, 'player_type': 'pitcher'})


if __name__ == "__main__":
    main()
