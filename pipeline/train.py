import copy
import datetime
import logging
import mlflow
import mlflow.keras
import pyarrow.feather as feather
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import Callback, CSVLogger
from tensorflow.data import Dataset, AUTOTUNE
import math
import numpy as np
import pandas as pd
import pathlib

from config import CONFIG, RELEVANT_SECTIONS
from hybrid.population_balance_model import ModelPBE

TF_FLOAT_DTYPE = tf.float32
TF_INT_DTYPE = tf.int32
NP_FLOAT_DTYPE = np.float32

# tf.compat.v1.enable_eager_execution()  # TODO: FOR TESTING ONLY - DISABLE!
# tf.config.run_functions_eagerly(True)  # TODO: FOR TESTING ONLY - DISABLE!
# tf.data.experimental.enable_debug_mode()  # TODO: FOR TESTING ONLY - DISABLE!

# Disable "Found untraced functions such as lstm_cell_layer_call_fn..." warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# tf.debugging.set_log_device_placement(True)  # To show in which device operations are done
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configure tensorflow
tf.get_logger().setLevel('ERROR')  # Reduce chatter
tf.compat.v1.ConfigProto().gpu_options.allow_growth = True

# Reproducible shuffling
tf.set_random_seed = 42
np.random.seed(42)


class CustomCallback(Callback):

    @staticmethod
    def on_epoch_end(epoch, logs=None):
        for metricName, metricValue in logs.items():
            mlflow.log_metric(metricName, metricValue, 1)

    # Very slow!
    # @staticmethod
    # def on_batch_end(batch, logs=None):
    #     for metricName, metricValue in logs.items():
    #         mlflow.log_metric(metricName, metricValue)


def create_tf_dataset(
        df_data: pd.DataFrame,
        features_history: list[str],
        features_prediction: list[str],
        features_controlled: list[str],
        nr_timesteps_history: int,
        nr_timesteps_prediction: int
) -> tf.data.Dataset:
    """
    Generates a tensorflow Dataset that can be used for training and evaluation. Slices the experiment data into windows
    and maps these to "history" and "controlled" model inputs as well as labels.
    :param pd.DataFrame df_data: Dataframe containing data for all features requested
    :param list[str] features_history: Features e.g. "sensor:intensity" as specified in config.json
    :param list[str] features_prediction: Features e.g. "sensor:intensity" as specified in config.json
    :param list[str] features_controlled: Features e.g. "sensor:intensity" as specified in config.json
    :param int nr_timesteps_history:
    :param int nr_timesteps_prediction:
    :return: tensorflow Dataset containing data windows for history, controlled and prediction / label.
    """
    if not isinstance(df_data, pd.DataFrame):
        raise TypeError(f'Expected type pd.DataFrame for df_data, got: {type(df_data)}.')
    if df_data.empty:
        raise ValueError(f'df_data must not be empty.')
    if not isinstance(features_history, list) or any(_ for _ in features_history if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_history, got: {type(features_history)}.')
    if not isinstance(features_prediction, list) or any(_ for _ in features_prediction if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_prediction, got: {type(features_prediction)}.')
    if not isinstance(features_controlled, list) or any(_ for _ in features_controlled if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_controlled, got: {type(features_controlled)}.')
    if not isinstance(nr_timesteps_history, int):
        raise TypeError(f'Expected type int for nr_timesteps_history, got: {type(nr_timesteps_history)}.')
    if not isinstance(nr_timesteps_prediction, int):
        raise TypeError(f'Expected type int for nr_timesteps_prediction, got: {type(nr_timesteps_prediction)}.')

    # Get dataframes for feature types
    time_index = df_data.index
    df_history = df_data[features_history]
    df_prediction = df_data[features_prediction]
    df_controlled = df_data[features_controlled]

    # Check if there is enough data
    if nr_timesteps_history + nr_timesteps_prediction > len(time_index):
        raise ValueError("nr_timesteps_history + nr_timesteps_prediction must be smaller than len(df_history.index)")

    # Lists with slice dataframe slices for all feature types
    list_slices_history, list_slices_prediction, list_slices_controlled = [], [], []
    start_index = nr_timesteps_history
    end_index = len(time_index) - nr_timesteps_prediction
    for j in range(start_index, end_index):
        list_slices_history.append(df_history[j-nr_timesteps_history:j])
        list_slices_prediction.append(df_prediction[j:j+nr_timesteps_prediction])
        list_slices_controlled.append(df_controlled[j:j+nr_timesteps_prediction]) if df_controlled is not None else []

    # Convert to np arrays
    array_slices_history = np.array(list_slices_history, dtype=NP_FLOAT_DTYPE)
    array_slices_prediction = np.array(list_slices_prediction, dtype=NP_FLOAT_DTYPE)
    array_slices_controlled = np.array(list_slices_controlled, dtype=NP_FLOAT_DTYPE)

    # TRAIN AND VAL DATA LENGTH MUST BE DIVISIBLE BY BATCH SIZE! -> https://stackoverflow.com/questions/64309194/invalidargumenterror-specified-a-list-with-shape-60-9-from-a-tensor-with-shap
    # Convert to tf dataset
    if array_slices_controlled.size != 0:
        tf_dataset = Dataset.from_tensor_slices((array_slices_history, array_slices_prediction, array_slices_controlled))
    else:
        tf_dataset = Dataset.from_tensor_slices((array_slices_history, array_slices_prediction))

    return tf_dataset


def prepare_tf_dataset(tf_dataset, buffer_size, batch_size):
    """
    Prepare tf Dataset for training by applying several operations on it:
        - cache: Cache the data in memory for quick access
        - shuffle: Generate randomized buffer dataset of buffer_size. buffer_size should be bigger than batch_size
        - map: Maps tensors of Dataset to model inputs "history" and "controlled", as well as labels / "prediction"
        - (repeat: Repeats dataset -> Doubles length)
        - batch: Splits randomized dataset into individual batches of batch_size
        - prefetch: Prefetches data while previous elements are still being processed. Should be at the end of a pipeline
    :param tensorflow Dataset tf_dataset:
    :param int buffer_size: Buffer size for shuffling
    :param int batch_size:
    :return: tensorflow Dataset
    """
    tf_dataset = tf_dataset \
        .cache() \
        .shuffle(buffer_size, reshuffle_each_iteration=True) \
        .map(lambda history, prediction, controlled: ({'history': history, 'controlled': controlled}, prediction)) \
        .batch(batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE) \
        .prefetch(AUTOTUNE)
    return tf_dataset


def compile_model(model, optimizer_name: str, optimizer_kwargs: dict, loss_function, metrics):
    # Create optimizer
    dict_tf_optimizers = {
        'Adadelta': optimizers.Adadelta,
        'Adagrad': optimizers.Adagrad,
        'Adam': optimizers.Adam,
        'Adamax': optimizers.Adamax,
        'Ftrl': optimizers.Ftrl,
        'SGD': optimizers.SGD,
        'Nadam': optimizers.Nadam,
        'RMSprop': optimizers.RMSprop
    }
    if optimizer_name in dict_tf_optimizers.keys():
        optimizer_class = dict_tf_optimizers.get(optimizer_name)
    else:
        raise ValueError(f'train.compile_model: Unknown optimizer name {optimizer_name}.')
    optimizer = optimizer_class(**optimizer_kwargs)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
        # run_eagerly=True  # TODO: ONLY FOR DEBUGGING! DISABLE!
    )


def train(config=CONFIG):

    # Import configuration
    features_history = config.get('features', {}).get('features_history')
    features_controlled = config.get('features', {}).get('features_controlled')
    features_prediction = config.get('features', {}).get('features_prediction')

    model_name = config.get('model', {}).get('model_name')
    model_type = config.get('model', {}).get('model_type')
    nr_timesteps_history = config.get('model', {}).get('nr_timesteps_history')
    nr_timesteps_prediction = config.get('model', {}).get('nr_timesteps_prediction')
    batch_size = config.get('model', {}).get('batch_size')

    validation_percentage = config.get('training', {}).get('validation_percentage')
    buffer_size = config.get('training', {}).get('buffer_size')
    nr_epochs = config.get('training', {}).get('nr_epochs')
    steps_per_epoch = config.get('training', {}).get('steps_per_epoch')
    loss_function = config.get('training', {}).get('loss_function')

    metrics = config.get('evaluation', {}).get('metrics')

    optimizer_name = config.get('optimizer', {}).get('optimizer_name')
    optimizer_kwargs = config.get('optimizer', {}).get('optimizer_kwargs')

    file_model_config = config.get('paths', {}).get('file_model_config')
    file_model_predict_config = config.get('paths', {}).get('file_model_predict_config')
    path_dataset = config.get('paths', {}).get('path_dataset')
    file_experiments = config.get('paths', {}).get('file_experiments')
    file_train_log = config.get('paths', {}).get('file_train_log')

    # Catch invalid parameters and cast types if necessary
    if not isinstance(model_name, str):
        raise TypeError(f'Expected type str for model_name, got: {type(model_name)}.')

    if not isinstance(model_type, str):
        raise TypeError(f'Expected type str for model_type, got: {type(model_type)}.')

    if isinstance(nr_timesteps_history, list):
        nr_timesteps_history = nr_timesteps_history[0]
    if not isinstance(nr_timesteps_history, int):
        raise TypeError(f'Expected type int for nr_timesteps_history, got: {type(nr_timesteps_history)}.')
    if nr_timesteps_history < 1:
        raise ValueError(f'Invalid value {nr_timesteps_history} for nr_timesteps_history given.')

    if isinstance(nr_timesteps_prediction, list):
        nr_timesteps_prediction = nr_timesteps_prediction[0]
    if not isinstance(nr_timesteps_prediction, int):
        raise TypeError(f'Expected type int for nr_timesteps_prediction, got: {type(nr_timesteps_prediction)}.')
    if nr_timesteps_prediction < 1:
        raise ValueError(f'Invalid value {nr_timesteps_prediction} for nr_timesteps_prediction given.')

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

    if isinstance(batch_size, list):
        batch_size = batch_size[0]
    if not isinstance(batch_size, int):
        raise TypeError(f'Expected type int for batch_size, got: {type(batch_size)}.')
    if batch_size < 1:
        raise ValueError(f'Invalid value {batch_size} for batch_size given.')

    if not isinstance(validation_percentage, int) and not isinstance(validation_percentage, float):
        raise TypeError(f'Expected type int or float for validation_percentage, got: {type(validation_percentage)}.')
    if not 0 <= validation_percentage <= 1:
        raise ValueError(f'Invalid value {validation_percentage} for validation_percentage given.')

    if isinstance(buffer_size, list):
        buffer_size = buffer_size[0]
    if not isinstance(buffer_size, int):
        raise TypeError(f'Expected type int for buffer_size, got: {type(buffer_size)}.')
    if buffer_size < 1:
        raise ValueError(f'Invalid value {buffer_size} for batch_size given.')

    if isinstance(nr_epochs, list):
        nr_epochs = nr_epochs[0]
    if not isinstance(nr_epochs, int):
        raise TypeError(f'Expected type int for nr_epochs, got: {type(nr_epochs)}.')
    if nr_epochs < 1:
        raise ValueError(f'Invalid value {nr_epochs} for nr_epochs given.')

    if isinstance(steps_per_epoch, list):
        steps_per_epoch = steps_per_epoch[0]
    if not isinstance(steps_per_epoch, int):
        raise TypeError(f'Expected type int for steps_per_epoch, got: {type(steps_per_epoch)}.')
    if nr_epochs < 0:
        raise ValueError(f'Invalid value {nr_epochs} for nr_epochs given.')
    if steps_per_epoch == 0:
        steps_per_epoch = None

    if isinstance(loss_function, list):
        loss_function = loss_function[0]
    if not isinstance(loss_function, str):
        raise TypeError(f'Expected type str for loss_function, got: {type(loss_function)}.')

    if not isinstance(metrics, list):
        raise TypeError(f'Expected type list[str] for metrics, got: {type(metrics)}.')
    if len(metrics) == 0:
        raise ValueError(f'Invalid value {metrics} for metrics given.')

    if isinstance(optimizer_name, list):
        optimizer_name = optimizer_name[0]
    if not isinstance(optimizer_name, str):
        raise TypeError(f'Expected type str for optimizer_name, got: {type(optimizer_name)}.')

    if not isinstance(optimizer_kwargs, dict):
        raise TypeError(f'Expected type dict for optimizer_kwargs, got: {type(optimizer_kwargs)}.')

    if not isinstance(file_model_config, pathlib.Path):
        raise TypeError(f'Expected type pathlib.Path for file_model_config, got: {type(file_model_config)}.')
    if not file_model_config.is_file():
        raise FileNotFoundError(f'No model config file found at {file_model_config}.')

    if not isinstance(file_model_predict_config, pathlib.Path):
        raise TypeError(f'Expected type pathlib.Path for file_model_predict_config, got: {type(file_model_predict_config)}.')
    if not file_model_predict_config.is_file():
        raise FileNotFoundError(f'No model predict config file found at {file_model_predict_config}.')

    if not isinstance(path_dataset, pathlib.Path):
        raise TypeError(f'Expected type pathlib.Path for path_dataset, got: {type(path_dataset)}.')
    if not path_dataset.is_dir():
        raise NotADirectoryError(f'No dataset directory found at {path_dataset}.')

    if not isinstance(file_experiments, pathlib.Path):
        raise TypeError(f'Expected type pathlib.Path for file_experiments, got: {type(file_experiments)}.')
    if not file_experiments.is_file():
        raise FileNotFoundError(f'Experiments file not found at {file_experiments}.')

    if not isinstance(file_train_log, pathlib.Path):
        raise TypeError(f'Expected type pathlib.Path for file_train_log, got: {type(file_train_log)}.')

    # Log relevant config params
    for section in RELEVANT_SECTIONS.get('train'):
        for key, val in config.get(section, {}).items():
            if len(str(val)) >= 250:
                continue
            mlflow.log_param(key, val)
    mlflow.log_artifact(str(file_experiments))

    # Get list of experiments
    with open(file_experiments, 'r') as f:
        list_experiments = f.readlines()
    list_experiments = [int(_.replace("\r", "").replace("\n", "")) for _ in list_experiments if _]

    # Shuffle experiments
    np.random.shuffle(list_experiments)

    # Load data for each experiment and create windows for training / validation
    nr_datasets_train = math.ceil(len(list_experiments) * (1 - validation_percentage))
    tf_dataset_train, tf_dataset_val = None, None

    for i, experiment_id in enumerate(list_experiments):

        # Load DataFrame
        path_dataset_experiment = path_dataset.joinpath(str(experiment_id)).absolute()
        try:
            df_data_experiment = feather.read_feather(str(path_dataset_experiment))
        except Exception as e:
            logging.error(f'train: No dataset found for experiment {experiment_id}. Skipping.')
            logging.warning(f'train: The following error was raised: {e}')
            continue

        # Create tf Dataset by slicing DataFrame into windows
        try:
            tf_dataset_experiment = create_tf_dataset(
                df_data_experiment,
                features_history,
                features_prediction,
                features_controlled,
                nr_timesteps_history,
                nr_timesteps_prediction
            )
        except Exception as e:
            logging.warning(f'train: No tf Dataset could be created for experiment {experiment_id}.')
            logging.warning(f'train: The following error was raised: {e}')
            continue

        # Split data for training and validation
        if i < nr_datasets_train:  # Add to training data
            tf_dataset_train = tf_dataset_train.concatenate(tf_dataset_experiment) \
                if tf_dataset_train is not None \
                else copy.copy(tf_dataset_experiment)
        else:  # Add to validation data
            tf_dataset_val = tf_dataset_val.concatenate(tf_dataset_experiment) \
                if tf_dataset_val is not None \
                else copy.copy(tf_dataset_experiment)

    # Generate batches
    if tf_dataset_train is None:
        raise ImportError('train: No training data was assembled. Aborting.')
    tf_dataset_train = prepare_tf_dataset(tf_dataset_train, buffer_size, batch_size)
    if tf_dataset_train is None:
        logging.warning('train: No validation data was assembled.')
    tf_dataset_val = prepare_tf_dataset(tf_dataset_val, buffer_size, batch_size) if tf_dataset_val else None

    # Create untrained model from config
    t0 = datetime.datetime.now()
    with open(file_model_config, "r") as r:
        model = model_from_json(
            r.read(),
            custom_objects={
                "ModelPBE": ModelPBE
            }
        )
    t1 = datetime.datetime.now()
    print(f'Loading model took {(t1 - t0).total_seconds()}s')

    # Compile model
    t0 = datetime.datetime.now()
    compile_model(
        model,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        loss_function=loss_function,
        metrics=metrics
    )
    t1 = datetime.datetime.now()
    print(f'Compiling model took {(t1 - t0).total_seconds()}s')

    # Train model
    csv_logger = CSVLogger(file_train_log, append=True, separator=';')
    t0 = datetime.datetime.now()
    model.fit(
        tf_dataset_train,
        epochs=nr_epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_data=tf_dataset_val,
        validation_steps=None,
        shuffle=False,
        callbacks=[
            CustomCallback(),
            csv_logger,
        ]
    )
    t1 = datetime.datetime.now()
    print(f'Training time: {(t1 - t0).total_seconds()} s')

    # Setup mlflow tracking for model
    mlflow.keras.log_model(
        model=model,
        artifact_path=model_name,
        registered_model_name=model_name
    )
    t2 = datetime.datetime.now()
    print(f'Log model time: {(t2 - t1).total_seconds()} s')

    # Create untrained model for prediction from config
    with open(file_model_predict_config, "r") as r:
        model_predict = model_from_json(r.read())

    # Transfer weights to model for prediction
    weights = model.get_weights()
    model_predict.set_weights(weights)
    t3 = datetime.datetime.now()
    print(f'Create prediction model time: {(t3 - t2).total_seconds()} s')

    # Setup mlflow tracking for model_predict
    mlflow.keras.log_model(
        model=model_predict,
        artifact_path=model_name + '_predict',
        registered_model_name=model_name + '_predict'
    )
    t4 = datetime.datetime.now()
    print(f'Log prediction model time: {(t4 - t3).total_seconds()} s')


if __name__ == '__main__':
    train()
