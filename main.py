import copy
import mlflow
from mlflow.tracking.fluent import _get_experiment_id
import optuna
import logging
import os

from config import CONFIG, PATH_CONFIG, VALID_TRIAL_PARAMETERS, VALID_TRIAL_CATEGORIES, print_config
from pipeline import already_ran
from pipeline.generate_dataset import generate_dataset
from pipeline.create_model import create_model
from pipeline.train import train


def run_pipeline(config: dict):
    mlflow_experiment_id = _get_experiment_id()
    logging.debug(f'MLflow experiment id: {mlflow_experiment_id}')

    logging.info('')
    entrypoint = 'generate_dataset'
    logging.info(f'==================== {entrypoint} ====================')
    logging.info('')
    generate_dataset_run = already_ran(entrypoint, config, mlflow_experiment_id=mlflow_experiment_id)
    if not generate_dataset_run:
        run_name = f'{entrypoint}:{config.get("dataset", {}).get("dataset_name")}'
        with mlflow.start_run(
                run_name=run_name,
                nested=True,
                tags={'entrypoint': entrypoint}
        ) as generate_dataset_run:
            generate_dataset(config)
    else:
        # Download dataset from mlflow artifact store
        try:
            mlflow.artifacts.download_artifacts(
                run_id=generate_dataset_run.info.run_id,
                dst_path=config.get('paths', {}).get('path_dataset')
            )
        except:
            logging.error(f"Couldn't download dataset artifacts for dataset {config.get('dataset', {}).get('dataset_name')}.")

    logging.info('')
    entrypoint = 'create_model'
    logging.info(f'==================== {entrypoint} ====================')
    logging.info('')
    create_model_run = already_ran(entrypoint, config, mlflow_experiment_id=mlflow_experiment_id)
    if not create_model_run:
        run_name = f'{entrypoint}:{config.get("model", {}).get("model_name")}'
        with mlflow.start_run(
                run_name=run_name,
                nested=True,
                tags={
                    'entrypoint': entrypoint,
                    'generate_dataset_run_id': generate_dataset_run.info.run_id,
                }
        ) as create_model_run:
            create_model(config)
    else:
        # Download model from mlflow artifact store
        file_model_config = config.get('paths', {}).get('file_model_config')
        if not file_model_config.is_file():
            try:
                mlflow.artifacts.download_artifacts(
                    run_id=create_model_run.info.run_id,
                    artifact_path=str(file_model_config),
                    dst_path=str(file_model_config)
                )
            except:
                logging.error(f"Couldn't download model file artifact {file_model_config}.")
        file_model_predict_config = config.get('paths', {}).get('file_model_predict_config')
        if not file_model_predict_config.is_file():
            try:
                mlflow.artifacts.download_artifacts(
                    run_id=create_model_run.info.run_id,
                    artifact_path=str(file_model_predict_config),
                    dst_path=str(file_model_predict_config)
                )
            except:
                logging.error(f"Couldn't download prediction model file artifact {file_model_predict_config}.")

    logging.info('')
    entrypoint = 'train'
    logging.info(f'==================== {entrypoint} ====================')
    logging.info('')
    train_run = already_ran(entrypoint, config, mlflow_experiment_id=mlflow_experiment_id)
    if not train_run:
        run_name = f'{entrypoint}:{config.get("model", {}).get("model_name")}'
        with mlflow.start_run(
                run_name=run_name,
                nested=True,
                tags={
                    'entrypoint': entrypoint,
                    'generate_dataset_run_id': generate_dataset_run.info.run_id,
                    'create_model_run_id': create_model_run.info.run_id
                }
        ) as train_run:
            train(config)

    metrics = train_run.data.metrics
    return metrics


def main():

    # Remove old train_log file
    file_train_log = CONFIG.get('paths', {}).get('file_train_log')
    if file_train_log.is_file():
        os.remove(file_train_log)

    def objective(trial):

        # Get a copy of the original CONFIG dict
        config_optuna = copy.deepcopy(CONFIG)
        print_config(config_optuna)

        # Let optuna make suggestions for parameters to optimize
        suggested_values = {category: {} for category in VALID_TRIAL_CATEGORIES}
        for category in VALID_TRIAL_CATEGORIES:
            for key, value in CONFIG.get(category, {}).items():
                if key in VALID_TRIAL_PARAMETERS and isinstance(value, list):

                    # Ignore parameters that normally are a list, if they don't contain lists
                    if key in [
                        'list_units_per_layer',
                        'list_dropout_per_layer'
                    ] and any([_ for _ in value if not isinstance(_, list)]):
                        continue

                    # Let optuna suggest values
                    if len(value) == 2 and all(isinstance(x, int) for x in value):
                        suggested_value = trial.suggest_int(key, min(value), max(value))
                        config_optuna[category][key] = suggested_value
                        suggested_values[category].update({key: suggested_value})
                    elif len(value) == 2 and all(isinstance(x, float) for x in value):
                        suggested_value = trial.suggest_float(key, min(value), max(value))
                        config_optuna[category][key] = suggested_value
                        suggested_values[category].update({key: suggested_value})
                    else:
                        suggested_value = trial.suggest_categorical(key, value)
                        config_optuna[category][key] = suggested_value
                        suggested_values[category].update({key: suggested_value})

                        # Legacy support
                        if key == "bias_regulizer":
                            config_optuna[category]["lstm_bias_regulizer"] = config_optuna[category][key]
                            config_optuna[category]["GRU_bias_regulizer"] = config_optuna[category][key]
                        elif key == "kernel_regulizer":
                            config_optuna[category]["lstm_kernel_regulizer"] = config_optuna[category][key]
                            config_optuna[category]["GRU_kernel_regulizer"] = config_optuna[category][key]

        # Run pipeline with values suggested by optuna
        print('Optuna suggested values:')
        print_config(suggested_values)
        metrics = run_pipeline(config_optuna)

        # Optimize on val_loss. If val_loss is empty, minimize normal loss
        obj_value = metrics.get('val_loss', metrics.get('loss'))

        # Return on invalid values for obj_value. This will enable the optuna study to continue
        # https://optuna.readthedocs.io/en/stable/faq.html#how-are-nans-returned-by-trials-handled
        if not isinstance(obj_value, float) or obj_value < 0.0:
            obj_value = float('nan')

        return obj_value

    # Start MLflow run
    run_name = f'main:{CONFIG.get("model", {}).get("model_name")}'
    logging.info(f'Starting MLflow run with name "{run_name}"')
    with mlflow.start_run(run_name=run_name):
        # Log config file
        mlflow.log_artifact(str(PATH_CONFIG))

        # Create and start optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=CONFIG.get('optuna', {}).get('nr_trials'))
        mlflow.log_dict(study.best_params, "study_best_params.json")
        logging.info(study.best_params)


if __name__ == '__main__':
    main()
