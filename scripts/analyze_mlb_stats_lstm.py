##########################################
## Script by Bradley T. Martin
##########################################

# import argparse
import datetime
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings(
    "ignore", message="Value in checkpoint could not be found in the restored"
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


from pybaseball import (
    batting_stats,
    pitching_stats,
    cache,
)

# Enable for faster data retrieval when running multiple times.
cache.disable()

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
from sklearn.base import BaseEstimator, TransformerMixin

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

from scikeras.wrappers import KerasRegressor


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """Class to force sklearn transformers to return pandas DataFrame instead of numpy array."""

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        X_transformed = self.transformer.transform(X)
        return pd.DataFrame(X_transformed, index=X.index, columns=X.columns)

    def inverse_transform(self, X):
        X_inv_transformed = self.transformer.inverse_transform(X)
        return pd.DataFrame(
            X_inv_transformed, index=X.index, columns=X.columns
        )


class Player:
    def __init__(
        self,
        player_data: pd.DataFrame,
        features: List[str],
        target_col: str,
        player_type: str,
        outdir: str,
        window_size=3,
        lower_is_better: List[str] = None,
        log_features: List[str] = None,
        train_months: List[int] = [4, 5, 6],
        target_months: List[int] = [7, 8, 9],
        target_season: int = 2023,
        sort_by_keys: List[str] = ["IDfg", "Season", "Month"],
        info_cols: List[str] = ["IDfg", "Name"],
        verbose=True,
    ) -> None:
        self.player_data = player_data.copy()
        self.features = features
        self.target_col = target_col
        self.player_type = player_type
        self.window_size = window_size
        self.lower_is_better = lower_is_better
        self.log_features = log_features
        self.outfile_prefix = os.path.join(outdir, self.player_type)
        self.train_months = train_months
        self.target_months = target_months
        self.target_season = target_season
        self.sort_by_keys = sort_by_keys
        self.info_cols = info_cols
        self.info_cols.append(self.target_col)
        self.verbose = verbose

        self.pred_data = None
        self.target_data = None
        self.train_data = None

        self.pred_ids = None
        self.target_ids = None
        self.train_ids = None
        self.feature_names_in = None
        self.scale_features = None
        self.to_drop = None

        self.imputer = DataFrameTransformer(SimpleImputer(strategy="mean"))
        self.log_scaler = DataFrameTransformer(
            PowerTransformer(method="box-cox")
        )
        self.feature_scaler = DataFrameTransformer(MinMaxScaler())
        self.target_scaler = MinMaxScaler()

    def preprocess_data(self):
        # Sort to make time sequences.
        self.player_data["Season"] = self.player_data["Season"].astype(int)
        self.player_data["Month"] = self.player_data["Month"].astype(int)
        self.player_data = self.player_data.sort_values(
            by=self.sort_by_keys, ascending=True
        )

        # Get current season data.
        self.pred_data = self.player_data[
            self.player_data["Season"] == self.target_season
        ].copy()

        self.pred_target = self.player_data[
            self.player_data["Season"] == self.target_season
        ]

        self.pred_target = self.pred_target[self.info_cols]

        self.pred_data = self.pred_data[
            self.pred_data["Month"].isin(self.train_months)
        ]

        # Get all previous season data, but not current season.
        self.player_data = self.player_data[
            self.player_data["Season"] != self.target_season
        ]

        self.train_data = self.player_data.copy()
        self.target_data = self.player_data.copy()

        self.pred_data = self.pred_data.sort_values(
            by=self.sort_by_keys, ascending=True
        )

        self.train_data = self.train_data.sort_values(
            by=self.sort_by_keys, ascending=True
        )

        self.target_data = self.target_data.sort_values(
            by=self.sort_by_keys, ascending=True
        )

        self.pred_ids = self.pred_data[self.info_cols]
        self.train_ids = self.train_data[self.info_cols]
        self.target_ids = self.target_data[self.info_cols]

        # Save output to CSV.
        self.train_data.to_csv(
            f"{self.outfile_prefix}_train_data.csv",
            header=True,
            index=False,
        )

        self.pred_data.to_csv(
            f"{self.outfile_prefix}_pred_data.csv",
            header=True,
            index=False,
        )

        self.target_data.to_csv(
            f"{self.outfile_prefix}_target_data.csv",
            header=True,
            index=False,
        )

        if self.verbose:
            print(f"Unique Pred IDs: {len(self.pred_ids['IDfg'].unique())}")
            print(f"Unique Train IDs: {len(self.train_ids['IDfg'].unique())}")
            print(
                f"Unique Target IDs: {len(self.target_ids['IDfg'].unique())}"
            )

    def get_features(self):
        self.to_drop = [
            "Age",
            "G",
            "AB",
            "PA",
            "IP",
        ]

        self.features = [x for x in self.features if x not in self.to_drop]
        self.feature_names_in = self.features

        self.features = [
            x
            for x in self.features
            if x not in ["IDfg", "Name", "Season", "Month", self.target_col]
        ]

        self.feature_names_out = self.features

        not_cols = ["IDfg", "Name", "Season", "Month", self.target_col]
        if self.log_features is not None:
            self.scale_features = [
                x
                for x in self.features
                if x not in self.log_features and x not in not_cols
            ]
        else:
            self.scale_features = [
                x for x in self.features if x not in not_cols
            ]

        self.necessary_columns = (
            self.info_cols
            + self.features
            + [self.target_col]
            + ["Season"]
            + ["Month"]
        )

    def get_features_target(self, training=False):
        if training:
            self.preprocess_data()

        X = self.train_data.copy() if training else self.pred_data.copy()
        y = self.target_data[self.info_cols].copy()

        self.get_features()
        X.drop(self.to_drop, axis=1, inplace=True, errors="ignore")
        X = X[self.necessary_columns]

        # Convert the relevant columns to numeric data type
        for feature in self.features:
            X[feature] = pd.to_numeric(X[feature], errors="coerce")
        if self.log_features is not None:
            X = X.loc[(X[self.log_features] > 0).all(axis=1)]

        if training:
            y = y.iloc[X.index]

        #### ONLY ON FEATURES, NOT TARGET ####
        # Rescales values where lower is better.
        for feature in self.features:
            if feature in self.lower_is_better:
                X[feature] = 1 / X[feature]

        if training:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            if self.log_features is not None:
                # Happens only on skewed features.
                self.log_scaler.fit(X_train[self.log_features])

            try:
                # Happens on all features.
                self.feature_scaler.fit(X_train[self.features])
            except ValueError:
                X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_train[self.features] = self.imputer.fit_transform(
                    X_train[self.features]
                )
                X_test[self.features] = self.imputer.transform(
                    X_test[self.features]
                )
                self.feature_scaler.fit(X_train[self.features])

            self.target_scaler.fit(
                y_train[self.target_col].values.reshape(-1, 1)
            )
            return X_train, X_test, y_train, y_test

        else:
            return X

    def pad_sequences_with_median(self, feature_data, window_size):
        # Calculate the median for each player
        player_medians = feature_data.groupby("IDfg").median()

        padded_data = []
        for player in feature_data["IDfg"].unique():
            player_data = feature_data[feature_data["IDfg"] == player]
            if len(player_data) < window_size:
                # If the player's data is less than the window size, pad it with the median
                pad_size = window_size - len(player_data)
                median_df = pd.DataFrame(
                    index=range(pad_size), columns=player_data.columns
                )
                for col in player_data.columns:
                    if col in player_medians.columns:
                        median_df[col] = player_medians.loc[player, col]
                # Append the original data and the padded data
                padded_player_data = pd.concat(
                    [player_data, median_df], ignore_index=True
                )
                padded_data.append(padded_player_data)
            else:
                # If the player's data is greater than or equal to the window size, keep it as is
                padded_data.append(player_data)

        # Concatenate all the padded data
        padded_feature_data = pd.concat(padded_data, ignore_index=True)

        return padded_feature_data

    def create_lstm_dataset(self, X, y=None):
        # Create sequences for each player
        sequences = []
        targets = []
        bad_idfgs = []
        names = []
        unscaled_target = []
        player_ids = []

        training = False if y is None else True

        if training:
            y[self.target_col] = self.target_scaler.transform(
                y[self.target_col].values.reshape(-1, 1)
            )
        else:
            y = self.pred_target.copy()

        for player in X["IDfg"].unique():
            # Get all data for given player.
            feature_data = X[X["IDfg"] == player].copy()
            target_data = y[y["IDfg"] == player].copy()

            if training:
                if len(feature_data) < self.window_size * 2:
                    continue
            else:
                if len(feature_data) < self.window_size:
                    bad_idfgs.append(player)

            if not training:
                player_ids.append(player)
                names.append(target_data["Name"].iloc[0])
                unscaled_target.append(target_data[self.target_col].mean())
            else:
                target_data = target_data.drop(["IDfg", "Name"], axis=1)

            # Split the data into first and second half
            feature_data_first_half = feature_data[
                feature_data["Month"].isin(self.train_months)
            ].copy()

            if training:
                feature_data_second_half = feature_data[
                    feature_data["Month"].isin(self.target_months)
                ].copy()

            if self.log_features is not None:
                # Transform per player.
                feature_data_first_half[
                    self.log_features
                ] = self.log_scaler.transform(
                    feature_data_first_half[self.log_features]
                )

            # if not training:
            # print(feature_data_first_half.describe())

            feature_data_first_half[
                self.features
            ] = self.feature_scaler.transform(
                feature_data_first_half[self.features]
            )

            # For the second half, if training, we will also want to transform this

            if self.log_features is not None:
                if training:
                    feature_data_second_half[
                        self.log_features
                    ] = self.log_scaler.transform(
                        feature_data_second_half[self.log_features]
                    )

            if training:
                feature_data_second_half[
                    self.features
                ] = self.feature_scaler.transform(
                    feature_data_second_half[self.features]
                )
            else:
                pad_feat = self.features + ["IDfg"]
                feature_data_first_half = self.pad_sequences_with_median(
                    feature_data_first_half[pad_feat], self.window_size
                )

            loop_range = (
                max(1, len(feature_data_first_half) - self.window_size)
                if not training
                else len(feature_data_first_half) - self.window_size
            )

            for i in range(loop_range):
                sequence = feature_data_first_half.iloc[
                    i : i + self.window_size
                ][self.features]

                if training:
                    start_index = i + 1
                    end_index = i + self.window_size + 1

                    if end_index > len(target_data):
                        continue

                    # Here we use the second half of the season data to generate targets
                    target = (
                        target_data[self.target_col]
                        .iloc[start_index:end_index]
                        .mean()
                    )

                    if np.isnan(target):
                        raise TypeError("NAN present in target sequence")

                    targets.append(target)
                if np.isnan(sequence.values).any():
                    raise TypeError("NAN present in sequences object.")
                sequences.append(sequence.values.tolist())

        if not training:
            self.pred_info = pd.DataFrame(
                {
                    "IDfg": player_ids,
                    "Name": names,
                    self.target_col: unscaled_target,
                }
            )

        sequences = np.array(sequences)

        if training:
            targets = np.array(targets)
        else:
            self.bad_idfgs = bad_idfgs

        if training:
            return sequences, targets
        else:
            return sequences


class RegressionCandidate:
    def __init__(
        self,
        players,
        window_size=3,
        param_grid=None,
        params=None,
        verbose=True,
    ):
        self.players = players
        self.player_model = None
        self.window_size = window_size
        self.verbose = verbose
        self.params = params

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

    def prepare_data(self, training=False):
        if training:
            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = self.players.get_features_target(
                training=True,
            )

            X_train, y_train = self.players.create_lstm_dataset(
                X_train, y=y_train
            )
            X_test, y_test = self.players.create_lstm_dataset(X_test, y=y_test)
            return X_train, X_test, y_train, y_test
        else:
            X = self.players.get_features_target(training=False)
            return self.players.create_lstm_dataset(X, y=None)

    def train_models(self):
        (
            X_train_player,
            X_test_player,
            y_train_player,
            y_test_player,
        ) = self.prepare_data(training=True)

        if self.verbose:
            print(f"Running {self.players.player_type} model...")

        (
            self.player_model,
            self.player_pipeline,
        ) = self._prepare_and_run_lstm_model(
            X_train_player,
            y_train_player,
            self.param_grid,
            self.players.player_type,
        )

        if self.verbose:
            print(
                f"Cross-validating holdout {self.players.player_type} test dataset..."
            )

        tensorboard_callback = self.player_pipeline
        callbacks = None if self.do_gridsearch else [tensorboard_callback]
        callbacks = [tensorboard_callback]
        self.player_model.model_.evaluate(
            X_test_player, y_test_player, callbacks=callbacks
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

        # ignore timesteps where all features are 0
        model.add(Masking(mask_value=0.0))

        # Add N layers depending on what user wants.
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
                print(f"Training {self.players.player_type} model...")

            log_dir = (
                f"results/logs_{self.players.player_type}/fit/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )

            tensorboard_callback = TensorBoard(
                log_dir=log_dir, histogram_freq=1
            )

            model = KerasRegressor(
                model_callable,
                loss="mse",
                metrics=["mae", "mse", "mape"],
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
                n_iter=800,
                scoring="neg_mean_squared_error",
                n_jobs=3,
                verbose=1,
                cv=3,
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
            print(
                f"DONE TRAINING {self.players.player_type.upper()} MODEL!\n\n"
            )
        return model, tensorboard_callback

    def predict_regression(self, model, X):
        X = X.copy()
        return np.squeeze(model.model_.predict(X))

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


def fit_predict(
    player_type,
    window_size,
    features,
    param_grid=None,
    params=None,
):
    target_col = "WAR" if player_type == "pitcher" else "wRC+"
    scrape_func = pitching_stats if player_type == "pitcher" else batting_stats
    stats_filename = f"results/stats_pickled/stats_{player_type}_fg.pkl"

    # Dowwnload from FanGraphs.
    df = load_or_scrape(stats_filename, scrape_func)

    # Account for features where lower is better.
    lower_is_better = (
        ["O-Swing%", "SwStr%"]
        if player_type == "hitter"
        else [
            "xFIP",
            "BABIP",
            "HardHit%",
            "WHIP",
        ]
    )

    log_features = (
        ["Barrel%", "BB/K", "O-Swing%"] if player_type == "hitter" else None
    )

    player = Player(
        df,
        features,
        target_col,
        player_type,
        "results/",
        window_size=window_size,
        lower_is_better=lower_is_better,
        log_features=log_features,
        train_months=[4, 5, 6],
        target_months=[7, 8, 9],
    )

    # Train the model.
    rc = RegressionCandidate(
        player,
        window_size=window_size,
        param_grid=param_grid,
        params=params,
        verbose=True,
    )
    rc.train_models()

    features = player.features
    model = rc.player_model
    output_filename = player.outfile_prefix + "_" + "pred.csv"

    X_pred = rc.prepare_data(training=False)
    predictions = rc.predict_regression(model, X_pred)
    predictions = player.target_scaler.inverse_transform(
        predictions.reshape(-1, 1)
    )

    predictions = np.squeeze(predictions)

    df_ids = player.pred_info.copy()
    df_ids = df_ids[~df_ids["IDfg"].isin(player.bad_idfgs)]

    predictions = pd.Series(predictions, name=f"{target_col}_Predictions")
    df_ids["Predictions"] = predictions
    df_ids["Predictions Difference"] = (
        df_ids["Predictions"] - df_ids[target_col]
    )

    df_ids = df_ids.sort_values(by=["Name"], ascending=True)

    df_ids.to_csv(output_filename, header=True, index=False)
    pred_plot(df_ids, player_type, target_col)
    plot_assess_model(df_ids, player_type, target_col)

    if player_type == "hitter":
        make_buncha_plots(df_ids, player_type, target_col)


def pred_plot(
    data, player_type, target_col, title_fontsize=32, label_fontsize=28
):
    fig, axs = plt.subplots(1, 2, figsize=(30, 60))

    value_ranges = [60, 75, 80, 100, 115, 140, 160]

    data = data.copy()
    # Sort the data by 'Name' column
    data = data.sort_values(by="Predictions Difference", ascending=False)

    # Normalize 'Predictions' column
    target_min = data[target_col].min()
    target_max = data[target_col].max()
    norm = plt.Normalize(target_min, target_max)

    seaborn_cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Apply color map to 'Predictions'
    colors = seaborn_cmap(norm(data[target_col]))

    xerr = []
    lower = [
        abs(min(x, 0)) for x in data["Predictions Difference"].values.tolist()
    ]
    upper = [
        abs(max(x, 0)) for x in data["Predictions Difference"].values.tolist()
    ]
    xerr.append(lower)
    xerr.append(upper)

    # Plot 1: Predictions
    axs[0].barh(data["Name"], data["Predictions"], color=colors, xerr=xerr)
    axs[0].axvline(0, color="black")
    axs[0].set_title(
        f"{player_type.capitalize()} {target_col} Predictions",
        fontsize=title_fontsize,
    )

    axs[0].set_xlabel(
        f"{target_col} Predictions (2nd Half)", fontsize=label_fontsize
    )
    axs[0].set_ylabel("Player Name", fontsize=label_fontsize)
    axs[0].tick_params(axis="both", labelsize=label_fontsize)

    # Plot 2: Predictions Difference
    axs[1].barh(data["Name"], data["Predictions Difference"], color=colors)
    axs[1].axvline(0, color="black")
    axs[1].set_title(
        f"{player_type.capitalize()} {target_col} Predictions Difference",
        fontsize=title_fontsize,
    )
    axs[1].set_xlabel(
        f"{target_col} Predictions Difference (2nd Half - First Half)",
        fontsize=label_fontsize,
    )
    axs[1].set_ylabel("")
    axs[1].tick_params(axis="both", labelsize=label_fontsize)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=seaborn_cmap),
        ax=axs[1],
        orientation="vertical",
        ticks=value_ranges,
        shrink=0.5,
        location="right",
    )

    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(label_fontsize)
    cbar.ax.set_title("wRC+ (1st Half)", fontdict={"fontsize": label_fontsize})

    fig.tight_layout()

    fig.savefig(
        f"results/pred_plot_{player_type}.png",
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close()


# Define the function to create "Rating" column
def rate_player(wrc):
    if 160 <= wrc:
        return "Excellent"
    elif 140 <= wrc < 160:
        return "Great"
    elif 115 <= wrc < 140:
        return "Above Average"
    elif 100 <= wrc < 115:
        return "Average"
    elif 80 <= wrc < 100:
        return "Below Average"
    elif 75 <= wrc < 80:
        return "Poor"
    else:
        return "Awful"


def make_buncha_plots(df, player_type, target_col):
    df = df.copy()
    # Create "Rating" column
    df["Rating"] = df[target_col].apply(rate_player)

    # Let's re-create the plots now
    # Plot 1: KDE plot of Predictions and wRC+
    fig, axs = plt.subplots(5, 1, figsize=(10, 50))

    sns.kdeplot(data=df, x="Predictions", fill=True, color="r", ax=axs[0])
    sns.kdeplot(data=df, x=target_col, fill=True, color="b", ax=axs[0])
    axs[0].set_title(f"KDE plot of Predictions and {target_col}")
    axs[0].legend(["Predictions", target_col])

    # Plot 2: Distribution of Predictions Difference across different rating categories
    sns.boxplot(
        data=df, y="Rating", x="Predictions Difference", orient="h", ax=axs[1]
    )
    axs[1].set_title("Boxplot of Predictions Difference across Ratings")

    # Plot 3: Scatter plot of Predictions vs wRC+ colored by Rating
    sns.scatterplot(
        data=df, x=target_col, y="Predictions", hue="Rating", ax=axs[2]
    )
    axs[2].set_title("Scatter plot of Predictions vs wRC+ colored by Rating")
    axs[2].legend(bbox_to_anchor=(1, 1))

    # Plot 4: Line plot of mean Predictions Difference across different rating categories
    mean_pred_diff = (
        df.groupby("Rating")["Predictions Difference"].mean().sort_values()
    )
    mean_pred_diff.plot(kind="barh", ax=axs[3])
    axs[3].set_title("Mean Predictions Difference across Ratings")

    # Plot 5: Violin plot of Predictions Difference across different rating categories
    sns.violinplot(
        data=df, y="Rating", x="Predictions Difference", orient="h", ax=axs[4]
    )
    axs[4].set_title("Violin plot of Predictions Difference across Ratings")

    fig.savefig(
        f"results/buncha_plots_{player_type}.png",
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close()


def plot_assess_model(df, player_type, target_col):
    fig, axs = plt.subplots(3, 1, figsize=(12, 24))

    # Histogram of Predictions Difference
    plt.sca(axs[0])
    sns.histplot(df["Predictions"], kde=True, color="skyblue")
    plt.title("Histogram of Predictions", fontsize=16)
    plt.xlabel("Predictions Difference", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    # Scatter plot of wRC+ vs Predictions
    plt.sca(axs[1])
    sns.scatterplot(x=target_col, y="Predictions", data=df, color="skyblue")
    plt.title(f"Scatter plot of {target_col} vs Predictions", fontsize=16)
    plt.xlabel(target_col, fontsize=14)
    plt.ylabel("Predictions", fontsize=14)
    plt.plot([0, 200], [0, 200], color="red")  # line y=x for reference

    # Boxplot of Predictions Difference
    plt.sca(axs[2])

    df_melt = pd.melt(df, id_vars=["IDfg", "Name"])
    sns.boxplot(x="variable", y="value", data=df_melt, color="skyblue")
    plt.title("Boxplot of Predictions", fontsize=16)
    plt.xlabel("Predictions", fontsize=14)

    fig.savefig(
        f"results/model_info_{player_type}.png",
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close()


def main():
    hitter_features = [
        "IDfg",
        "Name",
        "Season",
        "Month",
        # "AVG",
        "BABIP",
        "BB/K",
        "Barrel%",
        # "LD%",
        "O-Swing%",
        # "SwStr%",
        "wOBA",
        "wRC+",
    ]

    pitcher_features = [
        "IDfg",
        "Name",
        "Season",
        "Month",
        "xFIP",
        "BABIP",
        "HardHit%",
        "WHIP",
        "GB%",
        "K-BB%",
        "CSW%",
        "WAR",
    ]

    param_grid = {
        "model__neurons": [2, 3],
        "model__dropout": [0.0, 0.1, 0.2],
        "model__recurrent_dropout": [0.0, 0.1, 0.2],
        "optimizer__learning_rate": [0.1, 0.01, 0.001],
        "optimizer": ["adam"],
        "batch_size": [64],
        "model__n_layers": [1, 2],
        "model__kernel_regularizer": [0.0, 0.001, 0.01],
        "model__recurrent_regularizer": [0.0, 0.001, 0.01],
        "model__regularizer_type": ["L1L2"],
    }

    # Hitter params in KerasRegressor format with routing strings.
    hitter_params = {
        "optimizer__learning_rate": 0.1,
        "model__regularizer_type": "L1",
        "model__recurrent_regularizer": 0.0,
        "model__recurrent_dropout": 0.0,
        "model__neurons": 3,
        "model__n_layers": 1,
        "model__kernel_regularizer": 0.01,
        "model__dropout": 0.2,
        "batch_size": 64,
    }

    pitcher_params = {
        "optimizer__learning_rate": 0.1,
        "model__regularizer_type": "L1L2",
        "model__recurrent_regularizer": 0.01,
        "model__recurrent_dropout": 0.1,
        "model__neurons": 6,
        "model__n_layers": 2,
        "model__kernel_regularizer": 0.01,
        "model__dropout": 0.1,
        "batch_size": 32,
    }

    try:
        os.remove("results/train_test_data_hitter.pkl")
    except FileNotFoundError:
        pass

    try:
        os.remove("results/rc_hitter.pkl")
    except FileNotFoundError:
        pass

    try:
        os.remove("results/train_test_data_pitcher.pkl")
    except FileNotFoundError:
        pass

    try:
        os.remove("results/rc_pitcher.pkl")
    except FileNotFoundError:
        pass

    fit_predict("hitter", 3, hitter_features, param_grid, hitter_params)

    # fit_predict("pitcher", 3, pitcher_features, None, pitcher_params)


if __name__ == "__main__":
    main()
