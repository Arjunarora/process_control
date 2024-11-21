import copy
import json
import os
import pathlib
from hashlib import md5
import git
import mlflow
import logging
import sys

# Disable Influx warnings about using the pivot function
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)

# Disable unverified SSL cert notifications. The verification is disabled for MLflow, since it keeps failing.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PATH_CONFIG = pathlib.Path('./config.json').absolute()
if not PATH_CONFIG.is_file():
    raise FileNotFoundError(f'No config.json file was found at {PATH_CONFIG}. Please check working dir and existence of the file.')

FILE_FAULTY_EXPERIMENTS = pathlib.Path('faulty_experiments_list.txt').absolute()
MAX_EXPERIMENT_ID = 276  # todo: Update if necessary

CONFIG = {}

# Define dependencies between files and stages. E.g. if the code in generate_dataset.py has changed, cluster_experiments
# will also run again, since there may have been changes to the dataset used for training.
FILE_DEPENDENCIES = {
    'generate_dataset.py': [
        'generate_dataset',
        'create_model',
        'train'
    ],
    'create_model.py': [
        'create_model',
        'train'
    ],
    'train.py': [
        'train'
    ]
}

# Define sections of CONFIG that are relevant for entrypoint / pipeline step. E.g. if only section "dataset" is relevant
# for the pipeline step generate_dataset, it will ignore changes in sections model, training, etc. when checking if
# a run has already been done.
RELEVANT_SECTIONS = {
    'generate_dataset': ['dataset', 'features'],
    'generate_artificial_data': ['dataset'],
    'cluster_experiments': ['dataset'],
    'create_model': ['features', 'model'],
    # 'train': ['features', 'model', 'training', 'evaluation', 'optimizer']
    'train': ['dataset', 'features', 'model', 'training', 'evaluation', 'optimizer']  # Only for studys testing dataset param variations
}

VALID_TRIAL_PARAMETERS = [
    # dataset
    'nr_dps_per_minute',
    'nr_dps_smoothing',
    'min_nr_experiments',
    'max_nr_experiments',
    'nr_clusters',

    # model
    'model_type',  # new!
    'nr_timesteps_history',
    'nr_timesteps_prediction',
    'batch_size',
    'kernel_regulizer',
    'bias_regulizer',
    'dense_activation_function',
    'list_units_per_layer',

    # training
    'buffer_size',
    'nr_epochs',
    'steps_per_epoch',

    # optimizer
    'optimizer_name',
    'learning_rate',
    'epsilon',
    'beta_1',
    'beta_2',
    'amsgrad'
]

VALID_TRIAL_CATEGORIES = ["dataset", "model", "training", "optimizer", "evaluation"]


def init_config():

    # For getting output via docker logs
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.StreamHandler(sys.stderr))

    # Set experiment name
    repo = git.Repo('.')
    mlflow_experiment_name = repo.active_branch.name
    if not mlflow_experiment_name:
        mlflow_experiment_name = 'testing'
    mlflow.set_experiment(mlflow_experiment_name)


def print_config(config):
    for section in config.keys():
        print(f' {section}')
        for key, val in config[section].items():
            print(f'   {key}: {val}')


def extract_parameters(file_config):
    """
    This function extracts the global parameters required by pipeline code to run locally. Also used by main.py to check,
    if an identical MLflow run has already been run. It uses parameter instances from config.json file.
    Parameters from config.json will be automatically added to CONFIG, some are added or changed by this function.

    All parameter keys must be unique! E.g. if there are two parameters CONFIG["model"]["name"] and CONFIG["optimizer"]["name"],
    this will lead to problems in MLflow.

    """

    # Read the config file
    if file_config.is_file():
        with open(file_config, "r") as f:
            config = json.loads(f.read())

    # Automatically generate descriptors for dataset and model names

    # Check if list_units_per_layer contains lists itself (used for optuna)
    list_units_per_layer = config.get('model', {}).get('list_units_per_layer', [])
    if any([_ for _ in list_units_per_layer if isinstance(_, list)]):
        list_units_per_layer = list_units_per_layer[0]

    nr_layers = len(list_units_per_layer)
    nr_units = list_units_per_layer[-1] if list_units_per_layer else 0
    descriptor_layers = f'_{nr_layers}x{nr_units}'
    nr_timesteps_history = config.get("model", {}).get("nr_timesteps_history", 0)
    # --> If a pipeline step is called individually and a parameter is passed as a list (for optuna trials),
    # it will choose the first value of a list for naming models and datasets, since that the
    # pipeline functions will also use the first value of those lists when running.
    # --> If a pipeline step is called individually and a parameter is passed as a list (for optuna trials),
    nr_timesteps_prediction = config.get("model", {}).get("nr_timesteps_prediction", 0)
    descriptor_intervals = f'_{nr_timesteps_history[0] if isinstance(nr_timesteps_history, list) else nr_timesteps_history}' \
                           f'_{nr_timesteps_prediction[0] if isinstance(nr_timesteps_prediction, list) else nr_timesteps_prediction}'

    # Legacy: Old config.json files have features specified in model section
    if 'features' not in config.keys():
        config.update({'features': {
            'features_history': config.get('model', {}).get('features_history', []),
            'features_prediction': config.get('model', {}).get('features_prediction', []),
            'features_controlled': config.get('model', {}).get('features_controlled', []),
        }})
        config['model'].pop('features_history')
        config['model'].pop('features_prediction')
        config['model'].pop('features_controlled')

    # features

    features_history = config.get('features', {}).get('features_history', [])
    features_prediction = config.get('features', {}).get('features_prediction', [])
    features_controlled = config.get('features', {}).get('features_controlled', [])
    features_hybrid = config.get('features', {}).get('features_hybrid', [])
    config['features']['nr_features_history'] = len(features_history)
    config['features']['nr_features_prediction'] = len(features_prediction)
    config['features']['nr_features_controlled'] = len(features_controlled)
    config['features']['nr_features_hybrid'] = len(features_hybrid)

    descriptor_features = ''
    if 'sensor:intensity' in features_history:
        descriptor_features += '_S'
    if 'sensor:temperature' in features_history or 'thermostat:temperature_process' in features_history or 'ftir:temperature' in features_history:
        descriptor_features += '_T'
    if 'fbrm:counts_10' in features_history:
        descriptor_features += '_F10'
    if 'fbrm:counts_10-50' in features_history:
        descriptor_features += '_F10-50'
    if 'fbrm:counts_50-150' in features_history:
        descriptor_features += '_F50-150'
    if 'fbrm:counts_150-300' in features_history:
        descriptor_features += '_F150-300'
    if 'fbrm:counts_300-1000' in features_history:
        descriptor_features += '_F300-1000'
    if 'fbrm:counts_1000' in features_history:
        descriptor_features += '_F1000'
    if 'fe:nucleation_temperature' in features_history:
        descriptor_features += '_feNuclTemp'
    if 'fe:supersaturation' in features_history:
        descriptor_features += '_feS'
    nr_dps_per_minute = config.get("dataset", {}).get("nr_dps_per_minute", 0)
    nr_dps_smoothing = config.get("dataset", {}).get("nr_dps_smoothing", 0)
    descriptor_preprocessing = f'_{nr_dps_per_minute[0] if isinstance(nr_dps_per_minute, list) else nr_dps_per_minute}dpm' \
                              f'_{nr_dps_smoothing[0] if isinstance(nr_dps_smoothing, list) else nr_dps_smoothing}smooth'

    # dataset

    # Get actual list of experiments basis the parameter experiments.
    experiments = config.get('dataset', {}).get('experiments', [])
    list_experiments = []

    # List of experiment ids specified by user
    if isinstance(experiments, list):
        list_experiments = experiments

    # List containing all experiments
    elif isinstance(experiments, str) and experiments == 'all':
        # Get list of all experiments
        list_experiments = [int(_) for _ in range(MAX_EXPERIMENT_ID+1)]

    # List containing first batch of aa DNC experiments done in 2021
    elif isinstance(experiments, str) and experiments == 'first_batch':
        # Get list of all experiments
        list_experiments = [int(_) for _ in range(184+1)]

    # List containing second batch of aa / ga experiments done in 2022
    elif isinstance(experiments, str) and experiments == 'second_batch':
        # Get list of all experiments
        list_experiments = [int(_) for _ in range(185, 201+1)]

    # List containing third batch of aa / pdp DOE experiments done in 2023
    elif isinstance(experiments, str) and experiments == 'doe':
        # Get list of all experiments
        list_experiments = [int(_) for _ in range(202, 267+1)]

    # Ignore faulty experiments from the above list using a faulty experimemnts file
    with open(FILE_FAULTY_EXPERIMENTS, 'r') as r:
        faulty_experiments_raw = r.read()
        faulty_experiments = faulty_experiments_raw.split('\n')
        faulty_experiments = [int(_) for _ in faulty_experiments]
    list_experiments = sorted([x for x in list_experiments if x not in faulty_experiments])

    # Remove duplicates from the experiments list
    list_experiments = list(dict.fromkeys(list_experiments))

    # Legacy: In older versions experiment_threshold only specified the min value of experiment,
    # while the upper limit was always 50
    experiment_threshold = config['dataset'].get('experiment_threshold')
    min_nr_experiments = config['dataset'].get('min_nr_experiments')
    max_nr_experiments = config['dataset'].get('max_nr_experiments')
    if 'experiment_threshold' in config['dataset'].keys():
        if not min_nr_experiments:
            min_nr_experiments = experiment_threshold
        max_nr_experiments = 50
        config['dataset'].pop('experiment_threshold')
    config['dataset']['min_nr_experiments'] = min_nr_experiments
    config['dataset']['max_nr_experiments'] = max_nr_experiments

    # Legacy: old config.json versions had "nr_clusters" under "clustering" and not "dataset"
    if "clustering" in config.keys():
        config['dataset']['nr_clusters'] = config['clustering'].get('nr_clusters')

    # Legacy: Add values which were not present in previous config.json versions
    substance = config["dataset"].get("substance")
    density = config["dataset"].get("density")
    if not substance:
        logging.warning("No substance specified in config.json->dataset. Setting default value 'aa'.")
        substance = "aa"
    if not density:
        if substance == "aa":
            density = 1.37  # kg/dm^3
        elif substance == "pdp":
            density = 2.34  # kg/dm^3
    config['dataset']['substance'] = substance
    config['dataset']['density'] = density
    descriptor_substance = f'_{substance}' if substance else ''

    if not config["dataset"].get("experiment_threshold"):
        logging.warning("No experiment_threshold specified in config.json->dataset. Setting default value 18.")
        config['dataset']['experiment_threshold'] = 18
    if not config["dataset"].get("nr_clusters"):
        logging.warning("No nr_clusters specified in config.json->dataset. Setting default value 6.")
        config['dataset']['nr_clusters'] = 6

    dataset_name = f'{config.get("dataset", {}).get("dataset_name", "dataset")}{descriptor_preprocessing}{descriptor_intervals}{descriptor_features}{descriptor_substance}'
    config['dataset']['dataset_name'] = dataset_name

    # Insert default normalization ranges for legacy config.json files
    if not config['dataset'].get('normalization_ranges'):
        logging.warning('config.py: No normalization_ranges found in config file. Adding default ranges.')
        config['dataset']['normalization_ranges'] = {
            "ftir:temperature": [5, 75],
            "ftir:concentration": [0, 400],
            "fbrm:counts_10": [0, 300],
            "fbrm:counts_10-50": [0, 300],
            "fbrm:counts_10-100": [0, 300],
            "fbrm:counts_50-150": [0, 300],
            "fbrm:counts_150-300": [0, 300],
            "fbrm:counts_100-1000": [0, 300],
            "fbrm:counts_300-1000": [0, 300],
            "fbrm:counts_1000": [0, 300],
            "sensor:intensity": [0, 300],
            "sensor:temperature": [5, 75],
            "thermostat:temperature_process": [5, 75],
            "thermostat:temperature_internal": [5, 75],
            "thermostat:temperature_setpoint": [5, 75],
            "fe:supersaturation": [0, 5],
            "fe:nucleation_temperature": [5, 75]
        }

    if not config.get("dataset", {}).get("substance"):
        config["dataset"]["substance"] = "aa"

    # clustering

    # training

    # optimizer

    config['optimizer']['optimizer_kwargs'] = copy.deepcopy(config['optimizer'])
    config['optimizer']['optimizer_kwargs'].pop('optimizer_name')
    for key in config.get('optimizer', {}).get('optimizer_kwargs', {}):
        config['optimizer'].pop(key)

    # Legacy: Till now, kernel_regulizer and bias_regulizer were named after the used layers, e.g. lstm_kernel_regulizer.
    # To be able to compare with previous models, from now on all values are set if one is set.
    kernel_regulizer = config.get('model', {}).get('kernel_regulizer')
    if not kernel_regulizer:
        kernel_regulizer = config.get('model', {}).get('lstm_kernel_regulizer')
    if not kernel_regulizer:
        kernel_regulizer = config.get('model', {}).get('GRU_kernel_regulizer')
    config['model']['kernel_regulizer'] = kernel_regulizer
    config['model']['lstm_kernel_regulizer'] = kernel_regulizer
    config['model']['GRU_kernel_regulizer'] = kernel_regulizer
    bias_regulizer = config.get('model', {}).get('bias_regulizer')
    if not bias_regulizer:
        bias_regulizer = config.get('model', {}).get('lstm_bias_regulizer')
    if not bias_regulizer:
        bias_regulizer = config.get('model', {}).get('GRU_bias_regulizer')
    config['model']['bias_regulizer'] = bias_regulizer
    config['model']['lstm_bias_regulizer'] = bias_regulizer
    config['model']['GRU_bias_regulizer'] = bias_regulizer

    # model

    model_name = f'{config.get("model", {}).get("model_name", "model")}{descriptor_layers}{descriptor_intervals}{descriptor_features}{descriptor_substance}'
    model_type = config.get("model", {}).get("model_type", "lstm")
    config['model']['model_name'] = model_name
    config['model']['model_type'] = model_type

    # Legacy: old config.json versions had no "model_type" under "model"
    if not config["model"].get("model_type"):
        config['model']['model_type'] = "lstm"

    # evaluation

    # paths

    # Create paths if non-existant
    path_datasets = pathlib.Path(config.get('paths', {}).get('path_datasets', './datasets')).absolute()
    config['paths']['path_datasets'] = path_datasets
    if not path_datasets.is_dir():
        os.mkdir(path_datasets)

    path_dataset = path_datasets.joinpath(dataset_name).absolute()
    config['paths']['path_dataset'] = path_dataset
    if not path_dataset.is_dir():
        os.mkdir(path_dataset)
    config['paths']['file_experiments_original'] = path_dataset.joinpath('experiments_original.txt').absolute()
    config['paths']['file_experiments'] = path_dataset.joinpath('experiments.txt').absolute()

    path_clusters = path_dataset.joinpath("clusters")
    config['paths']['path_clusters'] = path_clusters
    if not path_clusters.is_dir():
        os.mkdir(path_clusters)

    path_models = pathlib.Path(config.get('paths', {}).get('path_models', './models')).absolute()
    config['paths']['path_models'] = path_models
    if not path_models.is_dir():
        os.mkdir(path_models)
    config['paths']['file_model_config'] = path_models.joinpath(f'{model_name}.json').absolute()
    config['paths']['file_model_predict_config'] = path_models.joinpath(f'{model_name}_predict.json').absolute()

    path_trained_models = pathlib.Path(config.get('paths', {}).get('path_trained_models', './trainedModels')).absolute()
    config['paths']['path_trained_models'] = path_trained_models
    if not path_trained_models.is_dir():
        os.mkdir(path_trained_models)
    config['paths']['file_trained_model'] = path_trained_models.joinpath(f'{model_name}.h5').absolute()
    config['paths']['file_trained_model_predict'] = path_trained_models.joinpath(f'{model_name}_predict.h5').absolute()

    config['paths']['file_train_log'] = path_trained_models.joinpath('train_log.csv').absolute()

    # other

    # Generate hash so changes in experiment id list can be detected
    # todo: Workaround. Ideally one would pass a list of experiments ids as parameter. That is currently not
    #   possible in mlflow. Passing it as string also fails because of the 250 char string parameter restriction
    str_experiments = '\n'.join([str(_) for _ in list_experiments])
    experiments_ver = md5(str_experiments.encode()).hexdigest()
    # Write all selected experiment IDs to file within dataset folder. The file will be logged by mlflow as artifact.
    with open(config.get('paths', {}).get('file_experiments_original'), 'w') as f:
        f.write(str_experiments)
    config['dataset']['experiments_ver'] = experiments_ver

    config.update({
        'mlflow': {
            'run_name': f'{dataset_name}->{model_name}'
        }
    })

    return config


# Init CONFIG on import
CONFIG = extract_parameters(PATH_CONFIG)
init_config()