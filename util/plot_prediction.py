import tensorflow as tf
from tensorflow.data.experimental import load as tf_load
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os
import shutil
import logging
import re
import shutil

import config  # Necessary for MLflow init by setting env vars etc.
from config import extract_parameters
from pipeline import already_ran
from hybrid.population_balance_model import PopulationBalanceModel


def user_select_registered_models() -> list[mlflow.entities.model_registry.registered_model.RegisteredModel] | list:
    """
    This function lets the user select registered models from the MLflow backend.
    :return:
    """
    list_selected_models = []

    # Get a sorted list of registered models
    client = mlflow.tracking.MlflowClient()
    list_models = client.search_registered_models(
        filter_string='name LIKE \'%_predict\'',
        max_results=1000,
        order_by=['timestamp ASC']
    )
    if not list_models:
        raise ValueError('No reqgistered models were returned from MLflow server.')
    list_model_names = [m.name for m in list_models]
    model_nr = len(list_model_names) - 1

    # User selection
    print('Select from the following model numbers ("all" to select all models):')
    for i, model_name in enumerate(list_model_names):
        print(f'{i}: {model_name}')
    while True:
        invalid_input = False
        user_input = input(f'Press ENTER to exit and use the newest model {model_nr}: {list_model_names[model_nr]}')
        if user_input == '':
            list_selected_models.append(list_models[-1])
        elif user_input == 'all':
            list_selected_models = list_models
        elif "," in user_input:
            list_user_input = user_input.strip(" ").split(",")
            for _ in list_user_input:
                try:
                    model_nr = int(_)
                    if model_nr in range(len(list_model_names)):
                        list_selected_models.append(list_models[model_nr])
                except:
                    invalid_input = True
                    break
        else:
            try:
                model_nr = int(user_input)
                if model_nr not in range(len(list_model_names)):
                    print('Model nr not found.')
                    continue
                list_selected_models = [list_models[model_nr]]
            except:
                invalid_input = True
        if invalid_input:
            print('Input must be an integer, a list of integers or "all".')
            continue
        break

    print(f'--> Selected model(s): {list_selected_models}')
    return list_selected_models


def user_select_registered_model_version(model_name: str) -> mlflow.entities.model_registry.registered_model.RegisteredModel | None:
    """
    Lets user select from model versions of a registered model.
    :param model_name:
    :return:
    """

    # Get list of registered model versions
    client = mlflow.tracking.MlflowClient()
    dict_registered_models = {_.version: _ for _ in list(reversed(client.search_model_versions(f"name='{model_name}'").to_list()))}
    if not dict_registered_models:
        print(f"No model versions returned from MLflow for model {model_name}")
        return
    selected_registered_model_version = list(dict_registered_models.keys())[-1]

    # User select
    print('Select from the following model versions:')
    print(f'{", ".join(_ for _ in dict_registered_models.keys())}')
    while True:
        user_input = input(f'Press ENTER to exit and use the newest model version: {selected_registered_model_version}')
        if user_input == "":
            break
        if user_input in dict_registered_models.keys():
            selected_registered_model_version = user_input
            break
        print('Invalid input.')
    print(f'--> Selected model version {selected_registered_model_version}')
    return dict_registered_models.get(selected_registered_model_version)


def user_select_experiments(model_run_id=None) -> list[int]:
    """
    This function asks user to select the experiments for which we want to download the data and plot predictions.
    :param model_run_id:
    :return:
    """
    if model_run_id:
        file_experiments = download_run_artifacts(model_run_id, 'experiments.txt')[0]
        with open(file_experiments, 'r') as f:
            list_experiments = f.readlines()
        list_experiments = [int(_.replace("\r", "").replace("\n", "")) for _ in list_experiments if _]
    else:
        list_experiments = [_ for _ in range(300)]
    list_selected_experiments = []

    print('Select from the following experiments ("all" for all experiments):')
    print(f'{", ".join([str(_) for _ in list_experiments])}')
    while True:
        user_input = input(f'Press ENTER to exit and use default experiment {list_experiments[-1]}')
        if user_input == '':
            list_selected_experiments.append(list_experiments[-1])
        elif user_input == 'all':
            list_selected_experiments = list_experiments
        else:
            matches = re.findall(r"(\d+)([ ]*-[ ]*\d+)?,?", user_input)
            if matches:
                for start, stop in matches:
                    start = int(start.replace(" ", ""))
                    if stop:
                        stop = int(stop.replace(" ", "").replace("-", ""))
                        list_selected_experiments.extend(list(range(start, stop)))
                    else:
                        list_selected_experiments.append(start)
                break
    print(f'--> Selected experiment(s): {list_selected_experiments}.')
    return list_selected_experiments


def download_run_artifacts(model_run_id: str, artifact_name="") -> list:
    """
    Downloads specified artifact(s) and returns a list of paths to the downloaded files.
    :param model_run_id:
    :param artifact_name:
    :return:
    """
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(model_run_id)
    path_temp = pathlib.Path('tmp').absolute()
    if not path_temp.is_dir():
        os.mkdir(path_temp)
    list_artifact_paths = []
    for artifact in artifacts:
        if (artifact_name and artifact.path == artifact_name) or not artifact_name:
            path_artifact = path_temp.joinpath(artifact_name).absolute()
            if path_artifact.is_dir():
                shutil.rmtree(path_artifact)
            elif path_artifact.is_file():
                os.remove(path_artifact)
            mlflow.artifacts.download_artifacts(
                run_id=model_run_id,
                artifact_path=artifact_name,
                dst_path=str(path_temp)
            )
            list_artifact_paths.append(path_artifact)
    return list_artifact_paths


@tf.function
def predict_with_model(model, array_history, array_controled, model_type):
    return model({'history': array_history, 'controlled': array_controled})


def load_dataset(path_dataset: pathlib.Path, config: dict) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads a dataset from specified path. Expects either a feahter file, or a tensorflow dataset, in which case it will
    reassemble data from saved windows.
    :param path_dataset:
    :param config:
    :return: Dataframes containing history and controlled data
    """

    # Get data from config.json of the parent run
    features_history = config.get('features', {}).get('features_history')
    features_controlled = config.get('features', {}).get('features_controlled')
    nr_features_history = config.get('features', {}).get('nr_features_history')
    nr_features_controlled = config.get('features', {}).get('nr_features_controlled')
    nr_timesteps_history = config.get('model', {}).get('nr_timesteps_history')
    nr_timesteps_prediction = config.get('model', {}).get('nr_timesteps_prediction')

    # Legacy tf.Dataset save format
    if path_dataset.is_dir():
        dataset_experiment = tf_load(str(path_dataset), compression="GZIP")
        if dataset_experiment is None:
            logging.error(f'Error loading data from {path_dataset}.')
            return None, None

        # Assemble complete history from slices
        list_slices = list(dataset_experiment.as_numpy_iterator())
        array_history = None
        for i in range(0, len(list_slices), nr_timesteps_history):
            array_history = np.append(array_history, list_slices[i][0]['history'], axis=0) \
                if array_history is not None \
                else list_slices[i][0]['history']
        diff_len = len(list_slices) % nr_timesteps_history
        if array_history is not None:
            array_history = np.append(array_history, list_slices[-1][0]['history'][-diff_len:], axis=0)
        array_history = np.append(array_history, [[np.nan for _ in range(nr_features_history)] for _ in range(nr_timesteps_prediction)], axis=0)
        df_history = pd.DataFrame(array_history, columns=features_history)

        # Assemble complete controlled timeseries from slices
        array_controlled = np.asarray([[np.nan for _ in range(nr_features_controlled)] for _ in range(nr_timesteps_history)])
        for i in range(0, len(list_slices), nr_timesteps_prediction):
            array_controlled = np.append(array_controlled, list_slices[i][0]['controlled'], axis=0)
        df_controlled = pd.DataFrame(array_controlled, columns=features_controlled)

    # Current feather save format
    else:
        df_experiment = pd.read_feather(path_dataset)
        if not isinstance(df_experiment, pd.DataFrame) or df_experiment.empty:
            logging.error(f'Error loading data from {path_dataset}.')
            return None, None
        df_history = df_experiment[features_history]
        df_controlled = df_experiment[features_controlled]

    return df_history, df_controlled


def make_prediction(model, df_history: pd.DataFrame, df_controlled: pd.DataFrame, config: dict) -> pd.DataFrame:

    # Get data from config.json of the parent run
    nr_dps_per_minute = config.get('dataset', {}).get('nr_dps_per_minute')
    substance = config.get('dataset', {}).get('substance')
    density = config.get('dataset', {}).get('density')

    features_history = config.get('features', {}).get('features_history')
    features_prediction = config.get('features', {}).get('features_prediction')
    features_controlled = config.get('features', {}).get('features_controlled')
    features_hybrid = config.get('features', {}).get('features_hybrid')

    nr_features_history = config.get('features', {}).get('nr_features_history')
    nr_features_prediction = config.get('features', {}).get('nr_features_prediction')
    nr_features_controlled = config.get('features', {}).get('nr_features_controlled')
    nr_features_hybrid = config.get('features', {}).get('nr_features_hybrid')

    nr_timesteps_history = config.get('model', {}).get('nr_timesteps_history')
    nr_timesteps_prediction = config.get('model', {}).get('nr_timesteps_prediction')
    nr_timesteps_controlled = nr_timesteps_prediction
    model_type = config.get('model', {}).get('model_type')

    h = config.get('model', {}).get('kwargs_hybrid', {}).get('h')
    l_max = config.get('model', {}).get('kwargs_hybrid', {}).get('l_max')
    l_critical = config.get('model', {}).get('kwargs_hybrid', {}).get('l_critical')
    k = config.get('model', {}).get('kwargs_hybrid', {}).get('k')

    pbe_model = None
    kwargs_pbe = {}
    if 'hybrid' in model_type:
        kwargs_pbe = {
            "features_history": features_history,
            "features_controlled": features_controlled,
            "features_prediction": features_prediction,
            "features_hybrid": features_hybrid,
            "nr_features_hybrid": nr_features_hybrid,
            "nr_dps_per_minute": nr_dps_per_minute,
            "nr_timesteps_history": nr_timesteps_history,
            "nr_timesteps_prediction": nr_timesteps_prediction,
            "h": h,
            "l_max": l_max,
            "l_critical": l_critical,
            "k": k,
            "substance": substance,
            "density": density
        }
        pbe_model = PopulationBalanceModel(
            **kwargs_pbe
        )

    # Prepare list of start indices for prediction
    list_indices = [_ for _ in range(nr_timesteps_history, df_history.shape[0] - nr_timesteps_prediction, 2 * nr_timesteps_prediction)]

    # Get predictions at different times in the experiment
    predictions = np.asarray([[np.nan for _ in range(nr_features_prediction)] for _1 in range(df_history.shape[0])], dtype=np.float32)
    for idx in list_indices:

        # Get slices of dfs for current prediction
        current_slice_history = df_history.iloc[idx - nr_timesteps_history:idx, :]
        current_slice_history = np.asarray([current_slice_history], dtype=np.float32)
        current_slice_controlled = df_controlled.iloc[idx:idx + nr_timesteps_prediction, :]
        current_slice_controlled = np.asarray([current_slice_controlled], dtype=np.float32)

        # Prevent edge cases where an array is empty
        if not current_slice_history.any() or not current_slice_controlled.any():
            continue

        # Prevent edge cases where an array doesn't have the correct shape
        if current_slice_history.shape != (1, nr_timesteps_history, nr_features_history) or current_slice_controlled.shape != (1, nr_timesteps_controlled, nr_features_controlled):
            continue

        # Predict!
        prediction_ann = predict_with_model(model, current_slice_history, current_slice_controlled, model_type).numpy()[0]
        if 'hybrid' in model_type:
            prediction = pbe_model.call([tf.convert_to_tensor(current_slice_history), tf.convert_to_tensor(current_slice_controlled), tf.convert_to_tensor([prediction_ann[0]])])
            prediction = prediction[0]
        else:
            prediction = prediction_ann

        predictions = np.insert(predictions, idx, prediction, axis=None if nr_features_prediction == 1 else 0)

    df_prediction = pd.DataFrame(predictions[:df_history.shape[0], :], columns=[_ + "_PREDICTED" for _ in features_prediction], index=df_history.index)

    return df_prediction


def plot_prediction():

    # Let user select registered models
    list_selected_registered_models = user_select_registered_models()
    if not list_selected_registered_models:
        raise ValueError("No registered models were selected.")

    # Let user select the experiments to be plotted (if possible)
    list_experiments = user_select_experiments()

    for registered_model in list_selected_registered_models:

        # If only one model was selected, let the user select the model version. Otherwise, just take the newest
        if len(list_selected_registered_models) == 1:
            registered_model_version = user_select_registered_model_version(registered_model.name)
        else:
            registered_model_version = registered_model.latest_versions[0]
        registered_model_run = mlflow.get_run(registered_model_version.run_id)
        registered_model_run_id = registered_model_run.info.run_id

        # Load the actual model
        logging.info(f'Loading model {registered_model_version.name} version {registered_model_version.version}')
        model = mlflow.tensorflow.load_model(f"models:/{registered_model_version.name}/{registered_model_version.version}")

        # Import configuration from the model run. If possible, get the run id of the corresponding generate_dataset run
        #   from tags - otherwise search for it with already_ran()
        model_parent_run = mlflow.MlflowClient().get_parent_run(registered_model_run_id)
        model_parent_run_id = model_parent_run.info.run_id
        file_config = download_run_artifacts(model_parent_run_id, 'config.json')[0]
        config_parent_run = extract_parameters(file_config)
        generate_dataset_run_id = registered_model_run.data.tags.get('generate_dataset_run_id')
        if not generate_dataset_run_id:
            generate_dataset_run_id = registered_model_run.data.tags.get('run_id_generate_dataset')
        if not generate_dataset_run_id:
            generate_dataset_run = already_ran('generate_dataset', config_parent_run, ignore_keys=['experiments_ver', 'normalization_ranges'], ignore_git=True)
            generate_dataset_run_id = generate_dataset_run.info.run_id
        if not generate_dataset_run_id:
            logging.error('No generate_dataset_run_id could be determined.')

        # Get data from config_parent_run.json of the parent run
        features_history = config_parent_run.get('features', {}).get('features_history')
        features_controlled = config_parent_run.get('features', {}).get('features_controlled')

        # Cycle through all selected experiments to be plotted
        for experiment_id in list_experiments:

            # Load dataset for current experiment
            list_artifact_paths = download_run_artifacts(generate_dataset_run_id, str(experiment_id))
            if not list_artifact_paths:
                logging.error(
                    f'plot_prediction: No artifact(s) returned for run {generate_dataset_run_id} for experiment {experiment_id}')
                continue
            path_dataset = list_artifact_paths[0]

            # Load dataset
            df_history, df_controlled = load_dataset(path_dataset, config_parent_run)

            # Make predictions
            df_prediction = make_prediction(model, df_history, df_controlled, config_parent_run)

            # Prepare df for plotting
            df_plot = pd.concat([
                    df_history,
                    df_controlled[[_ for _ in features_controlled if _ not in features_history]],
                    df_prediction
                ], axis=1,
                ignore_index=False
            ).reset_index(drop=True)

            # Determine paths for saving
            path_predictions = pathlib.Path('predictions').absolute()
            if not path_predictions.is_dir():
                os.mkdir(path_predictions)
            path_model = path_predictions.joinpath(f'{registered_model_version.name}-{registered_model_version.version}').absolute()

            if len(list_selected_registered_models) > 1 and len(list_experiments) == 1:
                path_experiment = path_predictions.joinpath(f'experiment_{experiment_id}').absolute()
                if not path_experiment.is_dir():
                    os.mkdir(path_experiment)
                file_img = path_experiment.joinpath(f'{registered_model_version.name}-{registered_model_version.version}.svg').absolute()
                file_xlsx = path_experiment.joinpath(f'models.xlsx').absolute()
                sheet_name_xlsx = f'{registered_model_version.name}-{registered_model_version.version}'
            else:
                if not path_model.is_dir():
                    os.mkdir(path_model)
                file_img = path_model.joinpath(f'experiment_{experiment_id}.svg').absolute()
                file_xlsx = path_model.joinpath(f'experiments.xlsx').absolute()
                sheet_name_xlsx = f'experiment_{experiment_id}'

            # Save the model
            path_model_predict = path_model.joinpath(f'model').absolute()
            if path_model_predict.is_dir():
                shutil.rmtree(path_model_predict)
            mlflow.tensorflow.save_model(model, path_model_predict)

            # Save data as excel table
            file_exists = file_xlsx.is_file()
            with pd.ExcelWriter(
                    file_xlsx,
                    mode="a" if file_exists else "w",
                    if_sheet_exists="replace" if file_exists else None,
                    date_format="YY-MM-DD hh:mm:ss.000"
            ) as writer:
                df_plot.to_excel(
                    writer,
                    sheet_name=sheet_name_xlsx,
                )

            # Create plot, save as image and show
            figure = df_plot.plot(figsize=(9.45*2, 4.72*2)).get_figure()
            figure.savefig(file_img, dpi=120)
            # plt.show()
            plt.close(figure)


if __name__ == '__main__':
    plot_prediction()
