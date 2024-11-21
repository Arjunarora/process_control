import datetime
import logging
import pyarrow.feather as feather
import mlflow
from mlflow.tracking.fluent import _get_experiment_id
import os
import pathlib
import pandas as pd
import numpy as np
import shutil
import re

from db import influx
from config import CONFIG, RELEVANT_SECTIONS
from pipeline import already_ran
import preprocessing.feature_engineering as fe
from preprocessing.ftir_concentration import get_supersaturation
from preprocessing.cluster_experiments import cluster_experiments
from preprocessing.generate_artificial_data import generate_artificial_data


def parse_feature(feature: str) -> (str, str):
    """
    Splits a feature specified in config.json in its parts device and channel. Device may be a device name,
    but also "fe" or "fix" to define engineered or fixed features.
    :param str feature: Feature string like "DEVICE:CHANNEL", "fe:FE_NAME", "fix:KEY=VALUE"
    :return: str device, str channel
    """
    if not isinstance(feature, str):
        raise TypeError(f'A feature to be parsed must be given as str. Received: {feature}')
    if re.match('^[a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+$', feature) is None:
        raise ValueError(f'Could not parse feature "{feature}" from config.')
    return feature.split(':')


def parse_features(features_history: list[str], features_prediction: list[str], features_controlled: list[str]) -> dict[str, list[str]]:
    """
    Creates a dict containing all features that will be used in the dataset. This contains features directly specified
    in config.json, as well as additional features needed for feature engineering / fixed features.
    :param list features_history:
    :param list features_prediction:
    :param list features_controlled:
    :return: A dict containing all features that will be used, e.g. {device: [channel1, channel2], "fix": ["a=100"], "fe": ["nucleation_temperature"]}
    """
    if not isinstance(features_history, list) or not isinstance(features_prediction, list) or not isinstance(features_controlled, list):
        raise TypeError('All feature lists must be given as lists of strings.')
    if any(_ for _ in features_history if not isinstance(_, str)) \
            or any(_ for _ in features_prediction if not isinstance(_, str)) \
            or any(_ for _ in features_controlled if not isinstance(_, str)):
        raise TypeError('All feature lists must be given as lists of strings.')

    dict_features_total = {}

    # Assemble list of all features specified in config.json
    for feature_list in [features_history, features_prediction, features_controlled]:
        for feature in feature_list:
            device, channel = parse_feature(feature)
            if device not in dict_features_total.keys():
                dict_features_total.update({device: [channel]})
            elif channel not in dict_features_total[device]:
                dict_features_total[device].append(channel)

    # Append devices / channels needed for feature engineering / fixed features, even if they will not directly be
    # used as features for training. E.g. if engineered feature "nucleation_time" is to be used for training, but
    # "sensor:intensity" is not, "sensor:intenstiy" must still be added because it is used for calculating the
    # nucleation time
    list_features_new = []
    for device, channels in dict_features_total.items():

        if device == 'fe':
            if 'nucleation_temperature' in channels:
                # For determination of the nucleation temperature sensor intensity and temperature are necessary to be
                # extracted from experiment data
                list_features_new.extend(["sensor:intensity", "sensor:temperature"])

            elif 'supersaturation' in channels:
                # For determination of the supersaturation, ftir concentration and temperature are necessary to be
                # extracted from experiment data
                list_features_new.extend(["ftir:concentration", "ftir:temperature"])

        if device == 'fix':
            pass  # Add fixed features depending on device measurements here

    # Update dict_features_total
    for feature in list_features_new:
        device, channel = parse_feature(feature)
        if device not in dict_features_total.keys():
            dict_features_total.update({device: [channel]})
        else:
            if channel not in dict_features_total.get(device):
                dict_features_total[device].append(channel)

    return dict_features_total


def get_experiment_data(experiment_id: int | str, dict_features_total: dict[str, list[str]], substance: str) -> pd.DataFrame:
    """
    Imports data of one experiment from influxdb. Which data is imported can be specified by dict_features_total.
    :param int experiment_id: Experiment ID as specified by "experiment_id" field tag in influx db
    :param dict dict_features_total: Dict of features to be imported. E.g. {'sensor': ['intensity', 'temperature']}
    :param str substance:
    :return: pd.DataFrame containing raw data from influxdb. Index: timestamps, col names: features
    """
    if not isinstance(substance, str):
        raise TypeError(f'Expected type str for substance, got: {type(substance)}.')
    if not isinstance(experiment_id, str) and not isinstance(experiment_id, int):
        raise TypeError('Experiment id must be of type int or str.')
    if not isinstance(dict_features_total, dict):
        raise TypeError('Argument dict_features_total must be of type dict.')

    df_experiment_data = pd.DataFrame({})

    for device, channels in dict_features_total.items():
        # Ignore engineered and fixed features
        if device in ['fe', 'fix']:
            continue

        # Get data for each channel and assemble in one DataFrame
        for channel in channels:
            df_channel = influx.get_experiment_timeseries(experiment_id, device, channel, substance)
            df_experiment_data = pd.concat([df_experiment_data, df_channel], axis=1, ignore_index=False)

    return df_experiment_data


def preprocess_data(
        df_data: pd.DataFrame,
        nr_dps_per_minute: int,
        nr_dps_smoothing: int,
        normalization_ranges: dict,
        do_remove_outliners=True,
        do_normalize=True,
        do_resample=True,
        do_smoothe=True
) -> pd.DataFrame:
    """
    Preprocesses a DataFrame by applying the following operations:
        - Aggregate datapoints by getting the mean value in time intervals defined by nr_dps_per_minute. This leads to
            every feature having the same amount of datapoints at the end.
        - Normalize the data in ranges specific to the features, so that all data fits in the interval [0, 1]
        - Smooth the data by applying rolling average with a window size of nr_dps_smooting.
    :param pd.Dataframe df_data: Dataframe containing timestamps as index, feature names as column names and feature
        data in the columns.
    :param int nr_dps_per_minute:
    :param int nr_dps_smoothing:
    :param dict normalization_ranges:
    :param do_remove_outliners:
    :param do_normalize:
    :param do_resample:
    :param do_smoothe:
    :return: pd.DataFrame
    """
    if not isinstance(df_data, pd.DataFrame):
        raise TypeError('Argument df_data must be provided as pd.DataFrame.')

    if isinstance(nr_dps_per_minute, list):
        nr_dps_per_minute = nr_dps_per_minute[0]
    if not isinstance(nr_dps_per_minute, int):
        raise TypeError(f'Expected type int for nr_dps_per_minute, got: {type(nr_dps_per_minute)}.')
    if nr_dps_per_minute < 1:
        raise ValueError(f'Invalid value {nr_dps_per_minute} for nr_dps_per_minute given.')

    if isinstance(nr_dps_smoothing, list):
        nr_dps_smoothing = nr_dps_smoothing[0]
    if not isinstance(nr_dps_smoothing, int):
        raise TypeError(f'Expected type int for nr_dps_smoothing, got: {type(nr_dps_smoothing)}.')
    if nr_dps_smoothing < 1:
        raise ValueError(f'Invalid value {nr_dps_smoothing} for nr_dps_smoothing given.')

    if not isinstance(normalization_ranges, dict):
        raise TypeError(f'Expected type dict for normalization_ranges, got: {type(normalization_ranges)}.')
    if any([_ for _ in normalization_ranges.values() if not isinstance(_, list)]):
        raise TypeError(f'Dict for normalization_ranges contains values of types other than list.')
    if any([_ for _, _1 in normalization_ranges.values() if not isinstance(_, int) or not isinstance(_1, int)]):
        raise TypeError(f'Normalization_ranges must be given as lists [int, int].')

    df_data_prep = pd.DataFrame({})

    # Determine start and end time of experiment and calculate nr of chunks.
    duration_experiment = df_data.index[-1] - df_data.index[0]
    duration_experiment_trimmed = duration_experiment - duration_experiment % datetime.timedelta(seconds=60)
    if duration_experiment_trimmed.total_seconds() > 172_800:  # Failsafe: Skip experiments longer than two days -> Probably faulty data!
        raise ValueError('Experiment data for more than two days found -> Faulty data?')
    nr_dps_dataset = int(duration_experiment_trimmed.total_seconds() / 60 * nr_dps_per_minute)
    chunk_length = duration_experiment_trimmed / nr_dps_dataset

    # Preprocess data for each feature
    for feature in df_data.columns:
        device, channel = parse_feature(feature)

        # Ignore engineered or fixed features
        if device in ['fe', 'fix']:
            continue

        # Determine limits for normalization depending on data type
        normalization_range = normalization_ranges.get(feature)
        if not normalization_range:
            raise ValueError(f"preprocess_data: No normalization_range found for feature {feature}. Aborting.")

        df_channel_prep = df_data[feature]

        # Remove values out of normalization range
        if do_remove_outliners:
            nr_nan_raw = df_channel_prep[df_channel_prep.isna()].size
            df_channel_prep = df_channel_prep.apply(lambda x: x if normalization_range[0] < x < normalization_range[1] else np.nan)
            nr_nan_outliners = df_channel_prep[df_channel_prep.isna()].size - nr_nan_raw
            percentage_outliners = nr_nan_outliners / df_channel_prep.size * 100
            if percentage_outliners > 10:
                logging.warning(f'generate_dataset.preprocess_data: {percentage_outliners} % of values were removed as outliners for feature {feature}.')

        # Aggregate datapoint mean over chunk intervals to get the same nr of datapoints for all features
        if do_resample:
            df_channel_prep = df_channel_prep.resample(chunk_length).mean()

        # Normalize dataframe and fill missing values
        if do_normalize:
            df_channel_prep = normalize_df(df_channel_prep, normalization_range).ffill().bfill()

        # Smooth dataframe
        if do_smoothe:
            df_channel_prep = df_channel_prep.rolling(nr_dps_smoothing, min_periods=1, center=True).mean()[:nr_dps_dataset]

        # Assemble dfs of normalized data
        df_data_prep = pd.concat([df_data_prep, df_channel_prep], axis=1, ignore_index=False)

    return df_data_prep


def normalize_df(df: pd.DataFrame | pd.Series, normalization_range: list[int | float] | tuple[int | float]) -> pd.DataFrame:
    """
    Normalizes all values within a dataframe
    :param df:
    :param normalization_range: Tuple containing min and max values in between which should be normalized.
    :return:
    """
    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series):
        raise TypeError('normalize_df: Argument df must be provided as pd.DataFrame or pd.Series.')
    if (not isinstance(normalization_range, list) and not isinstance(normalization_range, tuple)) or any([_ for _ in normalization_range if not isinstance(_, int) and not isinstance(_, float)]):
        print(f"DEBUG {type(df)=}")
        raise TypeError('normalize_df: Argument normalization_range must be provided as list[int | float] or tuple[int | float].')
    df_normalized = df.apply(lambda x: (x - normalization_range[0]) / (normalization_range[1] - normalization_range[0]))
    return df_normalized


def denormalize_df(df: pd.DataFrame | pd.Series, normalization_range: list[int | float] | tuple[int | float]) -> pd.DataFrame:
    """
    De-normalizes all values within a dataframe
    :param df:
    :param normalization_range: Tuple containing min and max values in between which should be normalized.
    :return:
    """
    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series):
        raise TypeError('denormalize_df: Argument df must be provided as pd.DataFrame or pd.Series.')
    if (not isinstance(normalization_range, list) and not isinstance(normalization_range, tuple)) or any([_ for _ in normalization_range if not isinstance(_, int) and not isinstance(_, float)]):
        raise TypeError('denormalize_df: Argument normalization_range must be provided as list[int | float] or tuple[int | float].')
    df_denormalized = df.apply(lambda x: x * (normalization_range[1] - normalization_range[0]) + normalization_range[0])
    return df_denormalized


def add_fixed_and_engineered_features(df_data: pd.DataFrame, dict_features_total: dict[str, list[str]], normalization_ranges: dict) -> pd.DataFrame:
    """
    Adds engineered and fixed value features to df_data according to dict_features_total.
    :param pd.DataFrame df_data: DataFrame containing preprocessed data
    :param dict dict_features_total: Dict containing all features
    :param dict normalization_ranges:
    :return: pd.DataFrame containing preprocessed data, engineered and fixed features
    """

    if not isinstance(df_data, pd.DataFrame):
        raise TypeError('Argument dict_data must be of type pd.DataFrame.')
    if not isinstance(dict_features_total, dict):
        raise TypeError('Argument dict_features_total must be of type dict.')

    if not isinstance(df_data, pd.DataFrame):
        raise TypeError('Argument df_data must be provided as pd.DataFrame.')

    if not isinstance(normalization_ranges, dict):
        raise TypeError(f'Expected type dict for normalization_ranges, got: {type(normalization_ranges)}.')
    if any([_ for _ in normalization_ranges.values() if not isinstance(_, list)]):
        raise TypeError(f'Dict for normalization_ranges contains values of types other than list.')
    if any([_ for _, _1 in normalization_ranges.values() if not isinstance(_, int) or not isinstance(_1, int)]):
        raise TypeError(f'Normalization_ranges must be given as lists [int, int].')

    df_data_fe = df_data.copy(deep=True)

    for device, channels in dict_features_total.items():

        if device not in ['fe', 'fix']:
            continue

        df_new = pd.DataFrame({})

        for channel in channels:
            feature = f'{device}:{channel}'

            # Add fixed value features
            if device == 'fix':
                value_pair = channel.split('=')
                if len(value_pair) != 2:
                    raise ValueError(
                        f'Fixed features must be specified as strings "fix:NAME=VALUE". Got {channel} instead of NAME=VALUE.')
                val = float(value_pair[1])
                df_new = pd.DataFrame(val, columns=[feature], index=df_data_fe.index)

            # Feature engineering
            elif device == 'fe':

                # Tries to determine the nucleation temperature based on features sensor:intensity and sensor:temperature.
                if channel == 'nucleation_temperature':
                    # Get sensor intensity data.
                    series_intensity = df_data_fe['sensor:intensity']
                    if series_intensity is None:
                        raise ValueError(
                            f'fe:nucleation_temperature: No data found for necessary feature sensor:intensity -> could not determine nucleation time. Aborting.')
                    # Get sensor temperature data.
                    series_temperature = df_data_fe['sensor:temperature']
                    if series_temperature is None:
                        raise ValueError(
                            f'fe:nucleation_temperature: No data found for necessary feature sensor:temperature -> could not determine nucleation time. Aborting.')
                    # Get normalized temperature at nucleation time
                    nucleation_time_index = fe.get_nucleation_time(
                        series_intensity=series_intensity,
                        nr_values_to_check=int(series_intensity.shape[0] / 50)
                    )
                    if nucleation_time_index not in series_temperature.index:
                        raise ValueError(
                            f'Nucleation time index "{nucleation_time_index}" not found in data for feature sensor:temperature.')
                    nucleation_temperature = series_temperature[nucleation_time_index]
                    df_new = pd.DataFrame(nucleation_temperature, columns=[feature], index=df_data_fe.index)

                # Calculates the supersaturation based on ftir:concentration and ftir:temperature
                # Usually, supersaturation should be availabe in influx as feature ftir:supersaturation. If so, this fe is not needed.
                elif channel == 'supersaturation':
                    # Get ftir concentration data.
                    df_concentration = df_data_fe[['ftir:concentration']]
                    df_concentration = denormalize_df(df_concentration, normalization_ranges.get("ftir:concentration"))
                    if not isinstance(df_concentration, pd.DataFrame) or df_concentration.empty:
                        raise ValueError(
                            f'fe:nucleation_temperature: No data found for necessary feature ftir:concentration -> could not determine supersaturation. Aborting.')
                    # Get ftir temperature data.
                    df_temperature = df_data_fe[['ftir:temperature']]
                    df_temperature = denormalize_df(df_temperature, normalization_ranges.get("ftir:temperature"))
                    if not isinstance(df_temperature, pd.DataFrame) or df_temperature.empty:
                        raise ValueError(
                            f'fe:nucleation_temperature: No data found for necessary feature ftir:temperature -> could not determine supersaturation. Aborting.')
                    # Get supersaturation
                    substance = config.get('dataset', {}).get('substance')
                    df_supersaturation = pd.DataFrame(
                        {
                            "fe:supersaturation": [
                                get_supersaturation(
                                    df_concentration.iloc[i, 0],
                                    df_temperature.iloc[i, 0],
                                    substance
                                ) for i in range(df_concentration.shape[0])
                            ]
                        },
                        index=df_data_fe.index
                    )
                    if df_supersaturation.empty:
                        raise ValueError(f'Supersaturation could not be calculated for feature fe:supersaturation.')
                    df_new = df_supersaturation

                else:
                    raise ValueError(f'Unknown engineered feature name "{channel}".')

            # Add new data
            df_data_fe = pd.concat([df_data_fe, df_new], axis=1, ignore_index=False)

    return df_data_fe


def generate_dataset(config=CONFIG):
    # Import configuration
    nr_dps_per_minute = config.get('dataset', {}).get('nr_dps_per_minute')
    nr_dps_smoothing = config.get('dataset', {}).get('nr_dps_smoothing')
    normalization_ranges = config.get('dataset', {}).get('normalization_ranges')
    substance = config.get('dataset', {}).get('substance')

    min_nr_experiments = config.get('dataset', {}).get('min_nr_experiments')
    max_nr_experiments = config.get('dataset', {}).get('max_nr_experiments')
    nr_clusters = config.get('dataset', {}).get('nr_clusters')
    art_data_method = config.get('dataset', {}).get('art_data_method')

    features_history = config.get('features', {}).get('features_history')
    features_prediction = config.get('features', {}).get('features_prediction')
    features_controlled = config.get('features', {}).get('features_controlled')

    path_dataset = config.get('paths', {}).get('path_dataset')
    file_experiments_original = config.get('paths', {}).get('file_experiments_original')
    file_experiments = config.get('paths', {}).get('file_experiments')

    # Catch invalid parameters and cast types if necessary
    if isinstance(nr_dps_per_minute, list):
        nr_dps_per_minute = nr_dps_per_minute[0]
    if not isinstance(nr_dps_per_minute, int):
        raise TypeError(f'Expected type int for nr_dps_per_minute, got: {type(nr_dps_per_minute)}.')
    if nr_dps_per_minute < 1:
        raise ValueError(f'Invalid value {nr_dps_per_minute} for nr_dps_per_minute given.')

    if isinstance(nr_dps_smoothing, list):
        nr_dps_smoothing = nr_dps_smoothing[0]
    if not isinstance(nr_dps_smoothing, int):
        raise TypeError(f'Expected type int for nr_dps_smoothing, got: {type(nr_dps_smoothing)}.')
    if nr_dps_smoothing < 1:
        raise ValueError(f'Invalid value {nr_dps_smoothing} for nr_dps_smoothing given.')

    if not isinstance(normalization_ranges, dict):
        raise TypeError(f'Expected type dict for normalization_ranges, got: {type(normalization_ranges)}.')
    if any([_ for _ in normalization_ranges.values() if not isinstance(_, list)]):
        raise TypeError(f'Dict for normalization_ranges contains values of types other than list.')
    if any([_ for _, _1 in normalization_ranges.values() if not isinstance(_, int) or not isinstance(_1, int)]):
        raise TypeError(f'Normalization_ranges must be given as lists [int, int].')

    if not isinstance(substance, str):
        raise TypeError(f'Expected type str for substance, got: {type(substance)}.')

    if isinstance(min_nr_experiments, list):
        min_nr_experiments = min_nr_experiments[0]
    if not isinstance(min_nr_experiments, int):
        raise TypeError(f'Expected type int for min_nr_experiments, got: {type(min_nr_experiments)}.')
    if min_nr_experiments < 0:
        raise ValueError(f'Invalid value {min_nr_experiments} for min_nr_experiments given.')

    if isinstance(max_nr_experiments, list):
        max_nr_experiments = max_nr_experiments[0]
    if not isinstance(max_nr_experiments, int):
        raise TypeError(f'Expected type int for max_nr_experiments, got: {type(max_nr_experiments)}.')
    if max_nr_experiments < 1:
        raise ValueError(f'Invalid value {max_nr_experiments} for max_nr_experiments given.')

    if isinstance(nr_clusters, list):
        nr_clusters = nr_clusters[0]
    if not isinstance(nr_clusters, int):
        raise TypeError(f'Expected type int for nr_clusters, got: {type(nr_clusters)}.')
    if nr_clusters < 1:
        raise ValueError(f'Invalid value {nr_clusters} for nr_clusters given.')

    if isinstance(art_data_method, list):
        art_data_method = art_data_method[0]
    if not isinstance(art_data_method, str):
        raise TypeError(f'Expected type int for art_data_method, got: {type(art_data_method)}.')
    if art_data_method not in ["jitter", "gan"]:
        raise ValueError(f'Invalid value {art_data_method} for art_data_method given.')

    if not isinstance(features_history, list) or any(_ for _ in features_history if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_history, got: {type(features_history)}.')
    if len(features_history) == 0:
        raise ValueError(f'Invalid value {features_history} for features_history given.')

    if not isinstance(features_prediction, list) or any(_ for _ in features_prediction if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_prediction, got: {type(features_prediction)}.')
    if len(features_prediction) == 0:
        raise ValueError(f'Invalid value {features_prediction} for features_prediction given.')

    if not isinstance(features_controlled, list) or any(_ for _ in features_controlled if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_controlled, got: {type(features_controlled)}.')
    if len(features_controlled) == 0:
        raise ValueError(f'Invalid value {features_controlled} for features_controlled given.')

    if not isinstance(path_dataset, pathlib.Path):
        raise TypeError(f'Expected type pathlib.Path for path_dataset, got: {type(path_dataset)}.')
    if not path_dataset.is_dir():
        raise NotADirectoryError(f'Dataset folder not found at {path_dataset}.')

    if not isinstance(file_experiments_original, pathlib.Path):
        raise TypeError(f'Expected type pathlib.Path for file_experiments_original, got: {type(file_experiments_original)}.')

    if not isinstance(file_experiments, pathlib.Path):
        raise TypeError(f'Expected type pathlib.Path for file_experiments, got: {type(file_experiments)}.')

    # Log relevant config params
    for section in RELEVANT_SECTIONS.get('generate_dataset'):
        for key, val in config.get(section, {}).items():
            if len(str(val)) >= 250:
                continue
            mlflow.log_param(key, val)

    # Get which data and which data columns will be used as features
    dict_features_total = parse_features(features_history, features_prediction, features_controlled)

    # Get list of experiments
    with open(file_experiments_original, 'r') as f:
        list_experiments_original = f.readlines()
    list_experiments_original = [int(_.replace("\r", "").replace("\n", "")) for _ in list_experiments_original if _]
    logging.debug(f'Original experiments to be used: {list_experiments_original}')

    # Log original experiments as an artifact
    mlflow.log_artifact(str(file_experiments_original))

    # Delete old data if necessary
    if path_dataset.is_dir():
        logging.debug(f'generate_dataset: Found existing directory at {path_dataset}. Deleting datasets.')
        for root, dirs, files in os.walk(path_dataset):  # Delete all folders
            for d in dirs:
                current_root = pathlib.Path(root).absolute()
                current_dir = current_root.joinpath(d).absolute()
                shutil.rmtree(current_dir)

    # Create new directory if necessary
    if not path_dataset.is_dir():
        os.mkdir(path_dataset)

    # Generate new training and validation datasets
    dict_dfs_original = {}
    for experiment_id in list_experiments_original:
        logging.info(f'generate_dataset: Processing experiment {experiment_id}.')
        mlflow.log_metric('processing_experiment', experiment_id)

        # Get experiment data from database server
        try:
            df_data_raw = get_experiment_data(
                experiment_id,
                dict_features_total,
                substance
            )
        except Exception as e:
            logging.warning(f'generate_dataset: No data could be extracted for experiment {experiment_id}.')
            logging.warning(f'generate_dataset: The following error was raised: {e}.')
            continue

        # Preprocess data
        try:
            df_data_prep = preprocess_data(
                df_data_raw,
                nr_dps_per_minute,
                nr_dps_smoothing,
                normalization_ranges,
                config
            )
        except Exception as e:
            logging.warning(f'generate_dataset: No data could be prepared for experiment {experiment_id}.')
            logging.warning(f'generate_dataset: The following error was raised: {e}')
            continue

        # Add fixed value and engineered features
        try:
            df_data_fe = add_fixed_and_engineered_features(
                df_data_prep,
                dict_features_total,
                normalization_ranges
            )
        except Exception as e:
            logging.warning(
                f'generate_dataset: Could not add fixed and engineered features for experiment {experiment_id}.')
            logging.warning(f'generate_dataset: The following error was raised: {e}')
            continue

        dict_dfs_original.update({experiment_id: df_data_fe})

    """
    Read path_dataset for the updated experiments list, and see how many experiments it originally has. 
    1. Implement the artificial data generator basis condition len_experiments <= 10.
    """
    nr_experiments = len(dict_dfs_original)

    # Check if the number of experiment files is below min_nr_experiments. If so: Generate artificial data.
    if nr_experiments < min_nr_experiments:

        # Determine the number of new synthetic experiments to create
        num_samples = min_nr_experiments - nr_experiments

        # Determine path where model for artificial data creation will be saved or loaded from
        file_model_artificial_data = None
        if art_data_method == 'gan':
            file_model_artificial_data = path_dataset.joinpath("model_gan")

        # Log entrypoint for the run
        entrypoint = 'generate_artificial_data'
        logging.info(f'==================== {entrypoint} ====================')
        logging.info('')

        generate_artificial_data_run = already_ran(entrypoint, config, mlflow_experiment_id=_get_experiment_id())
        if not generate_artificial_data_run:

            # Start an MLflow run for creating artificial data
            run_name = f'{entrypoint}:{config["dataset"]["dataset_name"]}'
            with mlflow.start_run(run_name=run_name, nested=True, tags={'entrypoint': entrypoint}):

                # Generate and save artificial data
                list_dfs_artificial = generate_artificial_data(
                    dict_dfs_original,
                    art_data_method=art_data_method,
                    num_samples=num_samples,
                    file_model_artificial_data=file_model_artificial_data,
                    nr_dps_smoothing=nr_dps_smoothing
                )

                # Save model_artificial_data to file and log with mlflow
                if file_model_artificial_data:
                    mlflow.log_artifact(str(file_model_artificial_data))

        else:

            # Download data from mlflow artifact store
            try:
                mlflow.artifacts.download_artifacts(
                    run_id=generate_artificial_data_run.info.run_id,
                    dst_path=str(path_dataset)
                )
            except:
                logging.error(
                    f"Couldn't download {entrypoint} artifacts for dataset {config.get('dataset', {}).get('dataset_name')}.")

            # Generate and save artificial data
            list_dfs_artificial = generate_artificial_data(
                dict_dfs_original,
                art_data_method=art_data_method,
                num_samples=num_samples,
                file_model_artificial_data=file_model_artificial_data,
                nr_dps_smoothing=nr_dps_smoothing
            )

        # Add artificial data to datasets for training, each df getting a unique artificial experiment id
        dict_dfs = dict_dfs_original.copy()
        dict_dfs.update({2000 + i: df_artificial for i, df_artificial in enumerate(list_dfs_artificial)})

    # Check if the number of experiment files is above max_nr_experiments. If so: Cluster data.
    elif nr_experiments > max_nr_experiments:

        # Log entrypoint for the run
        entrypoint = 'cluster_experiments'
        logging.info(f'==================== {entrypoint} ====================')
        logging.info('')

        # Start an MLflow run for clustering experiments
        run_name = f'{entrypoint}:{config["dataset"]["dataset_name"]}'
        with mlflow.start_run(run_name=run_name, nested=True, tags={'entrypoint': entrypoint}):
            nr_experiments_per_cluster = max(max_nr_experiments // nr_clusters, 1)  # Take only so many experiments per cluster that max_nr_experiments is reached
            dict_top_experiments, list_top_experiments, picture_clusters, silhouette_score_mean = cluster_experiments(
                dict_dfs_original,
                nr_clusters=nr_clusters,
                nr_experiments_per_cluster=nr_experiments_per_cluster
            )

            # Log interesting data to MLflow
            mlflow.log_metric('silhouette_score', silhouette_score_mean)
            mlflow.log_dict(dict_top_experiments, "top_experiments.json")
            mlflow.log_figure(picture_clusters, 'clusters.png')

            # Replace original dataset with shorter selection of experiments from clustering
            dict_dfs = {experiment_id: df for experiment_id, df in dict_dfs_original.items() if experiment_id in list_top_experiments}

    # Otherwise, if the amount of experiments is in the desired range between min_nr_experiments and max_nr_experiments,
    # use the original experiments.
    else:
        dict_dfs = dict_dfs_original

    # Log all clustered / artificial / original datasets to MLflow
    for experiment_id, df in (dict_dfs if len(dict_dfs) > len(dict_dfs_original) else dict_dfs_original).items():
        file_dataset_experiment = path_dataset.joinpath(str(experiment_id)).absolute()
        feather.write_feather(df, str(file_dataset_experiment))
        mlflow.log_artifact(str(file_dataset_experiment))

    # Compile experiments.txt containing the updated list of experiments and log to MLflow
    list_experiments = list(map(str, list(dict_dfs.keys())))
    experiment_list_str = '\n'.join(map(str, list_experiments))  # Convert int id to str
    with open(file_experiments, 'w') as f:
        f.write(experiment_list_str)
    mlflow.log_artifact(str(file_experiments))


if __name__ == '__main__':
    generate_dataset()
