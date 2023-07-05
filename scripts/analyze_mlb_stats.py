##########################################
## Script by Bradley T. Martin
##########################################

# import argparse
import os
import sys
from pathlib import Path
import datetime

# import requests
import pickle
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings(
    "ignore", message="Value in checkpoint could not be found in the restored"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pybaseball import (
    # statcast,
    # statcast_batter_expected_stats,
    # statcast_pitcher_expected_stats,
    # statcast_batter,
    # statcast_pitcher,
    batting_stats,
    # batting_stats_range,
    pitching_stats,
    # pitching_stats_range,
    # playerid_lookup,
    cache,
    # playerid_reverse_lookup,
)

# For faster data retrieval when running multiple times.
cache.disable()

# from feature_engine.selection import (
#     SmartCorrelatedSelection,
#     RecursiveFeatureElimination,
# )
from xgboost import XGBRegressor

# from feature_engine.selection import RecursiveFeatureElimination

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
)
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras import Input

from scikeras.wrappers import KerasRegressor

# from scipy.stats import zscore
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        X_transformed = self.transformer.transform(X)
        return pd.DataFrame(X_transformed, index=X.index, columns=X.columns)


class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_is_better=[], algorithm="pca", scaler=None):
        # a list of features where lower values are better
        self.lower_is_better = lower_is_better
        self.algorithm = algorithm  # pca, fa or composite
        self.scaler = scaler

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None

        if self.scaler is None:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)
        else:
            self.scaler_ = self.scaler
        return self

    def transform(self, X):
        # X = self.scaler_.transform(X)
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        if self.algorithm == "pca":
            return self.calculate_target_pca(X)
        elif self.algorithm == "fa":
            return self.calculate_target_fa(X)
        elif self.algorithm == "composite":
            return self.calculate_target_composite(X)
        elif self.algorithm == "ewm":
            return self.calculate_target_ewm(X)
        else:
            raise ValueError("Invalid algorithm specified")

    def calculate_target_ewm(self, X):
        for feature in X:
            # If lower values of the feature are better, invert it
            if feature in self.lower_is_better:
                X[feature] = X[feature].max() - X[feature]

        return X.ewm(span=3).mean().interpolate(method="linear")

    def calculate_target_pca(self, X):
        for feature in X:
            # If lower values of the feature are better, invert it
            if feature in self.lower_is_better:
                X[feature] = X[feature].max() - X[feature]

        # Apply PCA and get the first principal component
        pca = PCA(n_components=1)

        Xt = pca.fit_transform(X)
        if len(Xt.shape) > 1:
            Xt = np.squeeze(Xt)
        return Xt

    def calculate_target_fa(self, X):
        for feature in X:
            # If lower values of the feature are better, invert it
            if feature in self.lower_is_better:
                X[feature] = X[feature].max() - X[feature]

        # Apply Factor Analysis and get the first factor
        fa = FactorAnalysis(n_components=1)
        Xt = fa.fit_transform(X)
        if len(Xt.shape) > 1:
            Xt = np.squeeze(Xt)
        return Xt

    def calculate_target_composite(self, X):
        for feature in X:
            # If lower values of the feature are better, invert it
            if feature in self.lower_is_better:
                X[feature] = X[feature].max() - X[feature]

        max_silhouette = -1
        best_n_clusters = 2

        # Try different numbers of clusters and keep the one with the highest silhouette score
        for n_clusters in range(2, 11):  # Try from 2 to 10 clusters
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, labels)

            if silhouette_avg > max_silhouette:
                max_silhouette = silhouette_avg
                best_n_clusters = n_clusters

        # Perform K-Means clustering with the best number of clusters
        kmeans = KMeans(n_clusters=best_n_clusters)
        y = kmeans.fit_predict(X)

        # Fit the Random Forest model and get feature importances
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = model.feature_importances_

        # Calculate the composite score as a weighted sum of the features
        return X.dot(importances)


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
        params=None,
        features=None,
        lstm=False,
        verbose=True,
    ):
        self.player_data = player_data.copy()
        self.player_type = player_type
        self.player_model = None
        self.window_size = window_size
        self.features = features
        self.lstm = lstm
        self.verbose = verbose
        self.params = params

        self.feature_scaler = DataFrameTransformer(StandardScaler())

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

            if params is None:
                raise TypeError(
                    "params must not be NoneType of param_grid is NoneType"
                )
        else:
            self.do_gridsearch = True
        self.param_grid = param_grid

    # def calculate_target(self, sequence, features):
    #     # Calculate the z-scores for each feature
    #     for feature in features:
    #         if (
    #             feature != "SkippedMonth"
    #         ):  # Don't standardize the indicator variable
    #             if sequence[feature].std() != 0:
    #                 sequence[feature] = zscore(sequence[feature])
    #             else:
    #                 sequence[feature] = 0  # or any other default value

    #     # Define features where a higher value is worse for performance
    #     negative_impact_features_hitters = ["O-Swing%", "SwStr%"]
    #     negative_impact_features_pitchers = ["AVG", "WHIP", "BABIP", "Barrel%"]

    #     # Check if these features exist in your data and adjust their sign
    #     for feature in negative_impact_features_hitters:
    #         if feature in sequence.columns:
    #             sequence[feature] = -sequence[feature]

    #     for feature in negative_impact_features_pitchers:
    #         if feature in sequence.columns:
    #             sequence[feature] = -sequence[feature]

    #     # Sum the z-scores to create the 'target' value
    #     target = sequence[features].sum(axis=1).mean()

    #     return target

    def get_train_test_lstm(
        self, data, features, window_size=2, training=False
    ):
        data = data.copy()
        data = data.sort_values(by=["IDfg", "Season", "Month"], ascending=True)

        features = [
            x for x in features if x not in ["IDfg", "Name", "Season", "Month"]
        ]

        self.feature_names_out = features

        # Convert the relevant columns to numeric data type
        for feature in features:
            data[feature] = pd.to_numeric(data[feature], errors="coerce")
        data = data.loc[(data[features] > 0).all(axis=1)]

        # Impute missing data.
        self.imputer = DataFrameTransformer(SimpleImputer(strategy="mean"))
        data[features] = self.imputer.fit_transform(data[features])

        # Feature scaling
        if training:
            data[features] = self.feature_scaler.fit_transform(data[features])
        else:
            data[features] = self.feature_scaler.transform(data[features])

        # Account for features where lower is better.
        lower_is_better = (
            ["O-Swing%", "SwStr%"]
            if self.player_type == "hitter"
            else ["AVG", "WHIP", "BABIP", "Barrel%"]
        )

        # Create sequences for each player
        sequences = []
        targets = []
        bad_idfgs = []

        max_sequence_length = window_size  # the length of the full sequence

        for player in data["IDfg"].unique():
            player_data = data[data["IDfg"] == player]

            if len(player_data) < window_size:
                bad_idfgs.append(player)
                continue

            tt = TargetTransformer(
                lower_is_better=lower_is_better,
                algorithm="ewm",
                scaler=self.feature_scaler,
            )

            # Include the "IDfg" column when fitting and transforming with the TargetTransformer
            tt.fit(player_data[features])

            for i in range(len(player_data) - window_size + 1):
                sequence = player_data.iloc[i : i + window_size][features]
                next_sequence = player_data.iloc[i + 1 : i + 1 + window_size][
                    features
                ]

                # # Add 'MissedTime' feature to the sequence
                # missed_months = max_sequence_length - len(next_sequence)
                # sequence = sequence.assign(MissedTime=missed_months)

                # if next_sequence is shorter than max_sequence_length, pad it
                if len(next_sequence) < max_sequence_length:
                    padding = pd.DataFrame(
                        -1,
                        index=np.arange(
                            max_sequence_length - len(next_sequence)
                        ),
                        columns=next_sequence.columns,
                    )
                    next_sequence = pd.concat([next_sequence, padding])

                target = tt.transform(next_sequence).mean().tolist()

                # target = next_sequence.mean().mean() - sequence.mean().mean()
                sequences.append(sequence.values)
                targets.append(target)

        idfg = data[["IDfg", "Name"]]
        idfg = idfg[~idfg["IDfg"].isin(bad_idfgs)]
        idfg = idfg.drop_duplicates(subset=["IDfg"], ignore_index=True)

        X = np.array(sequences)
        y = np.array(targets)

        if training:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        else:
            return X, y, idfg

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

    def prepare_data(self, data, features, training=False, lstm=False):
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

    def train_models(self, lstm=False):
        func = (
            self._prepare_and_run_lstm_model
            if lstm
            else self._prepare_and_run_model
        )

        if Path(f"results/train_test_data_{self.player_type}.pkl").is_file():
            with open(
                f"results/train_test_data_{self.player_type}.pkl", "rb"
            ) as fin:
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
                self.player_data, self.features, training=True, lstm=lstm
            )

            with open(
                f"results/train_test_data_{self.player_type}.pkl", "wb"
            ) as fout:
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

        if self.verbose:
            print(
                f"Cross-validating holdout {self.player_type} test dataset..."
            )

        if not self.lstm:
            self.xval(X_test_player, y_test_player, cv=5)
        else:
            if self.do_gridsearch:
                self.player_model.model_.evaluate(
                    X_test_player,
                    y_test_player,
                )
            else:
                # Dumb way of doing it but I'm being lazy.
                tensorboard_callback = self.player_pipeline
                self.player_model.model_.evaluate(
                    X_test_player,
                    y_test_player,
                    callbacks=[tensorboard_callback],
                )

    def model_build_fn(compile_kwargs, **kwargs):
        window_size = kwargs.get("window_size")
        n_features = kwargs.get("n_features")
        dropout = kwargs.get("dropout", 0.2)
        recurrent_dropout = kwargs.get("recurrent_dropout", 0.2)
        neurons = kwargs.get("neurons", (n_features // 2) + 1)
        n_layers = kwargs.get("n_layers", 1)
        kernel_regularizer = kwargs.get("kernel_regularizer", 0.0)
        recurrent_regularizer = kwargs.get("recurrent_regularizer", 0.0)
        regularizer_type = kwargs.get("regularizer_type", "L1")
        regularizer_type = regularizer_type.upper()

        if regularizer_type == "L1":
            kern_reg = L1(kernel_regularizer)
            rec_reg = L1(recurrent_regularizer)
        elif regularizer_type == "L2":
            kern_reg = L2(kernel_regularizer)
            rec_reg = L2(recurrent_regularizer)
        elif regularizer_type == "L1L2":
            kern_reg = L1L2(l1=kernel_regularizer, l2=kernel_regularizer)
            rec_reg = L1L2(l1=recurrent_regularizer, l2=kernel_regularizer)
        else:
            raise ValueError(
                f"Invalid regularizer_type provided: {regularizer_type}. Expected 'L1', 'L2', or 'L1L2'"
            )

        model = Sequential()
        model.add(Input(shape=(window_size, n_features)))

        for i in range(n_layers):
            if i == n_layers - 1:
                return_sequences = False
            else:
                return_sequences = True
            model.add(
                LSTM(
                    neurons,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=kern_reg,
                    recurrent_regularizer=rec_reg,
                )
            )

        model.add(Dense(n_features))
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
        early_stopping_callback = EarlyStopping(patience=5)
        tensorboard_callback = None
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

            log_dir = (
                f"results/logs_{self.player_type}/fit/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )

            tensorboard_callback = TensorBoard(
                log_dir=log_dir, histogram_freq=1
            )

            model = KerasRegressor(
                model_callable,
                loss="mse",
                metrics="mse",
                validation_split=0.2,
                optimizer=Adam,
                epochs=100,
                **self.params,
                verbose=0,
                callbacks=[early_stopping_callback, tensorboard_callback],
            )
            model.fit(X_train, y_train)
        else:
            if self.verbose:
                print(f"Doing {player_type} grid search...")
            grid_search = RandomizedSearchCV(
                KerasRegressor(
                    model=model_callable,
                    loss="mse",
                    verbose=0,
                    epochs=100,
                    validation_split=0.2,
                    callbacks=[early_stopping_callback],
                    **additional_params,
                ),
                param_distributions=param_grid,
                n_iter=1000,
                scoring="neg_mean_squared_error",
                n_jobs=3,
                verbose=1,
                cv=5,
            )

            grid_search.fit(X_train, y_train)

        self.n_features = X_train.shape[1]

        if self.do_gridsearch:
            model = grid_search.best_estimator_
            if self.verbose:
                print(
                    f"Best {player_type} grid search score: {grid_search.best_score_}"
                )
                print(f"Best {player_type} params: {grid_search.best_params_}")

            self.plot_grid_search_results(grid_search, player_type)

        if self.verbose:
            print(f"DONE TRAINING {player_type.upper()} MODEL!\n\n")
        return model, tensorboard_callback

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
                    cv=5,
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
                    f"Best {player_type} grid search score: {grid_search.best_score_}"
                )
                print(f"Best {player_type} params: {grid_search.best_params_}")

            self.plot_grid_search_results(grid_search, player_type)
        self.plot_feature_importances(model, X_train, player_type)

        if self.verbose:
            print(f"DONE TRAINING {player_type.upper()} MODEL!\n\n")
        return model, pipeline

    def predict_regression(self, model, X):
        X = X.copy()
        if not self.lstm:
            if "IDfg" in X.columns:
                X = X.drop(["IDfg"], axis=1)
            if "target" in X.columns:
                X = X.drop(["target"], axis=1)
            return model.predict(X)
        else:
            return np.squeeze(model.model_.predict(X))

    def xval(self, X_test, y_test, cv=5):
        scores = cross_val_score(
            self.player_model,
            X_test,
            y_test,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
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
        params = list(grid_search.best_params_.keys())
        params.sort()
        results = grid_search.cv_results_
        outfile = f"results/gridsearch_results_{player_type}.png"
        fig, axs = plt.subplots(
            nrows=len(params), figsize=(10, 5 * len(params))
        )

        for ax, param in zip(axs, params):
            sns.boxplot(
                x=f"param_{param}", y="mean_test_score", data=results, ax=ax
            )
            ax.set_title(f"Mean test scores for different values of {param}")

        plt.tight_layout()
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

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
    filename,
    df,
    player_type,
    features,
    window_size,
    param_grid,
    params,
    lstm=True,
):
    if Path(filename).is_file() and not lstm:
        with open(filename, "rb") as fin:
            rc = pickle.load(fin)
    else:
        rc = RegressionCandidate(
            df,
            player_type,
            window_size=window_size,
            param_grid=param_grid,
            params=params,
            features=features,
            lstm=lstm,
            verbose=True,
        )
        rc.train_models(lstm=lstm)

    if not lstm:
        with open(filename, "wb") as fout:
            pickle.dump(rc, fout)
    return rc


def preprocess_data(df, pipeline, rc=None, lstm=False):
    if lstm and rc is None:
        raise TypeError("rc cannot be NoneType if lstm is True")

    if lstm:
        return df
    else:
        transformers = pipeline[:-1]
        for t in transformers:
            df = t.transform(df)
        return df


def run_pipeline(
    player_type,
    window_size,
    features,
    param_grid=None,
    params=None,
    lstm=False,
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
            df, 2023, [4, 5, 6], ["IDfg", "Season", "Month"], player_type
        )
        rc = load_or_train_model(
            rc_filename,
            df,
            player_type,
            features,
            window_size,
            param_grid,
            params,
            lstm=lstm,
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
        # df = make_features(
        #     df,
        #     feature_cols_to_drop,
        #     diff_feature1,
        #     diff_feature2,
        #     new_feature_name,
        # )
        df, df_current_season = process_data(
            df, 2023, [4, 5, 6], ["IDfg", "Season", "Month"], player_type
        )

        # df.to_csv("data/hitter_stats.csv", header=True, index=False)

        # df_current_season.to_csv(
        #     "data/hitter_stats_pred.csv", header=True, index=False
        # )

        rc = load_or_train_model(
            rc_filename,
            df,
            player_type,
            features,
            window_size,
            param_grid,
            params,
            lstm=lstm,
        )
        features = rc.features

        pipeline = rc.player_pipeline
        model = rc.player_model
        output_filename = "results/hitter_pred.csv"

    df_current_season, _, df_ids = rc.prepare_data(
        df_current_season, features, lstm=lstm
    )

    df_current_season = preprocess_data(
        df_current_season, pipeline, rc=rc, lstm=lstm
    )

    predictions = rc.predict_regression(model, df_current_season)

    df_pred = pd.DataFrame(predictions, columns=rc.feature_names_out)
    df_pred["Predictions"] = df_pred.mean(axis=1)

    df_ids = pd.concat([df_ids, df_pred["Predictions"]], axis=1)
    df_ids = df_ids.sort_values(by=["Predictions"])
    df_ids.to_csv(output_filename, header=True, index=False)

    plot_data(df_ids, player_type)


def plot_data(data, player_type):
    fig, ax = plt.subplots(1, 1, figsize=(20, 30))

    plt.sca(ax)
    sns.barplot(x="Predictions", y="Name", data=data, orient="h")
    plt.axvline(0, color="red")
    plt.title(f"{player_type.capitalize()} Regression Predictions")
    plt.xlabel("Prediction (Std Dev)")
    plt.ylabel("Name")
    fig.savefig(
        f"results/pred_plot_{player_type}.png",
        facecolor="white",
    )
    plt.close()


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
        "model__neurons": [6, 8],
        "model__dropout": [0.1, 0.2],
        "model__recurrent_dropout": [0.1, 0.2],
        "optimizer__learning_rate": [0.1, 0.01, 0.001],
        "optimizer": ["adam"],
        "batch_size": [64, 128],
        "model__n_layers": [1, 2, 3, 4],
        "model__kernel_regularizer": [0.0, 0.001, 0.01],
        "model__recurrent_regularizer": [0.0, 0.001, 0.01],
        "model__regularizer_type": ["L1L2"],
    }

    # hitter_params = {
    #     "colsample_bytree": 0.5,
    #     "learning_rate": 0.2,
    #     "max_depth": 2,
    #     "n_estimators": 50,
    #     "reg_alpha": 0.1,
    #     "reg_lambda": 0.0001,
    # }

    hitter_params = {
        "model__neurons": 6,
        "model__dropout": 0.1,
        "model__recurrent_dropout": 0.1,
        "optimizer__learning_rate": 0.001,
        "batch_size": 64,
        "model__kernel_regularizer": 0.0,
        "model__recurrent_regularizer": 0.0,
        "model__regularizer_type": "L1",
        "model__n_layers": 2,
    }

    run_pipeline(
        "hitter", 3, hitter_features, param_grid, hitter_params, lstm=True
    )

    # run_pipeline("pitcher", 2, pitcher_features, param_grid, lstm=False)


# # Perform a Marcel Projection for a specific hitter
# hitter_projection = rc.marcel_projection({'player_id': 1, 'age': 25, 'player_type': 'hitter'})

# # Perform a Marcel Projection for a specific pitcher
# pitcher_projection = rc.marcel_projection({'player_id': 2, 'age': 30, 'player_type': 'pitcher'})


if __name__ == "__main__":
    main()
