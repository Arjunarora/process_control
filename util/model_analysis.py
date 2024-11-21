import mlflow
import pandas as pd
import logging


def mlflow_all_registered_models():
    """
    This function creates a list of all the registered models .
    :return:
    """
    client = mlflow.tracking.MlflowClient()
    list_models_sorted = [model for model in client.list_registered_models(max_results=1000) if "_predict" in model.name]  # todo: replace with search_experiments(). list_registered_models will be depricated.
    list_models_sorted.sort(key=lambda x: x.last_updated_timestamp)
    dict_registered_models = {model.name: model for model in list_models_sorted}
    if not dict_registered_models:
        logging.error('plot_prediction.user_select_model: No registered models retrieved from MLflow. Aborting')
        return
    list_models = list(dict_registered_models.values())
    list_models.sort(key=lambda x: x.last_updated_timestamp)
    list_model_names = [m.name for m in list_models]
    return list_model_names

def model_versions_list(model_list):
    client = mlflow.tracking.MlflowClient()
    dict_info = {'creation_timestamp': [], 'model_name': [], 'version': [], 'run_id': [], 'source': [], 'status': [],
                 'tags': []}
    for model_name in model_list:
        list_model_version_objects = client.search_model_versions(f"name='{model_name}'")
        for version in range(len(list_model_version_objects)):
            dict_info['creation_timestamp'].append(list_model_version_objects[version].creation_timestamp)
            dict_info['model_name'].append(list_model_version_objects[version].name)
            dict_info['version'].append(list_model_version_objects[version].version)
            dict_info['run_id'].append(list_model_version_objects[version].run_id)
            dict_info['source'].append(list_model_version_objects[version].source)
            dict_info['status'].append(list_model_version_objects[version].status)
            dict_info['tags'].append(list_model_version_objects[version].tags)

    return dict_info

def download_params_metrics(run_id):
    client = mlflow.tracking.MlflowClient()
    model_run = mlflow.get_run(run_id)
    param_dict = model_run.data.params
    metric_dict = model_run.data.metrics
    return param_dict | metric_dict

#SO I need a function which will list all model names, then model versions.
#Once I have a nested list of model, model_versions, I can start download params and metrics corresponding to those model_versions.
#And then I will have a database of sorts which I can easily analyse.
def model_analysis():
    """
    all_model_list = mlflow_all_registered_models()

    dc = model_versions_list(all_model_list)
    model_df = pd.DataFrame.from_dict(dc, orient='index').transpose()
    """
    model_df = pd.read_csv("model_data.csv")

    data_df = pd.DataFrame()
    for run_id in model_df['run_id']:
        data_dict = download_params_metrics(run_id)
        data_df = data_df.append(data_dict,ignore_index=True)

    data_df.to_csv("run_data.csv")


if __name__ == '__main__':
    model_analysis()

