import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import pyarrow.feather as feather
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics import silhouette_samples, silhouette_score


def cluster_experiments(dict_dfs: dict[str: pd.DataFrame], nr_clusters=10, nr_experiments_per_cluster=1):

    # Select clustering temperature feature
    feature_clustering = None
    df_sample = list(dict_dfs.values())[0]
    if 'thermostat:temperature_process' in df_sample.columns:
        feature_clustering = 'thermostat:temperature_process'
    elif 'sensor:temperature' in df_sample.columns:
        feature_clustering = 'sensor:temperature'

    # Scale and resample each dataframe and add padding
    nr_timesteps_resampled = 50
    list_arrays = [np.expand_dims(df[feature_clustering].to_numpy(), 0) for df in dict_dfs.values()]
    list_arrays_resampled = [TimeSeriesResampler(sz=nr_timesteps_resampled).fit_transform(_) for _ in list_arrays]
    X = np.concatenate(list_arrays_resampled, axis=0)

    # Run DTW-based K-means clustering to group similar experiments
    km_dtw = TimeSeriesKMeans(n_clusters=nr_clusters, metric='softdtw', random_state=0, max_iter=50).fit(X)
    labels = km_dtw.labels_

    # Calculate mean silhouette score
    silhouette_score_mean = silhouette_score(X[:, :, 0], km_dtw.fit_predict(X[:, :, 0]))

    # Create figure of clusters
    plt.figure()
    if nr_clusters <= 9:
        cols, rows = 3, 3
    elif nr_clusters <= 16:
        cols, rows = 4, 4
    else:
        cols, rows = 5, 5

    for i_cluster in range(nr_clusters):
        plt.subplot(rows, cols, i_cluster + 1)
        # Add lines from other clusters first...
        for i_data, array_data in enumerate(X):
            if labels[i_data] != i_cluster:
                plt.plot(array_data.ravel(), color='silver', linestyle='-')
        # ...then data belonging to the cluster...
        for i_data, array_data in enumerate(X):
            if labels[i_data] == i_cluster:
                plt.plot(array_data.ravel(), color='black', linestyle='-')
        # ...and lastly the cluster mean
        plt.plot(km_dtw.cluster_centers_[i_cluster].ravel(), "r-")
    figure_clusters = plt.gcf()
    plt.show()

    # Create dictionary with cluster labels and corresponding experiment names
    dict_clusters = {i_cluster: {} for i_cluster in range(nr_clusters)}
    for i_df, i_cluster in enumerate(labels):
        experiment_id, df = list(dict_dfs.items())[i_df]
        dict_clusters[i_cluster].update({experiment_id: df})

    dict_experiment_stats = {}
    dict_top_experiments = {}  # Contains experiments, scores and dfs of all clusters
    list_top_experiments = []  # Contains list of only the selected experiment_ids

    for i_cluster, dict_cluster in dict_clusters.items():
        cluster_experiments_scored = []
        cluster_stats = None

        if not dict_cluster:
            dict_top_experiments[i_cluster] = {}
            continue

        # Calculate the descriptive statistics of the whole cluster based on the temperature
        for experiment_id, df in dict_cluster.items():
            df_experiment_stats = df.reset_index()[feature_clustering].describe()
            dict_experiment_stats.update({experiment_id: df_experiment_stats})

            cluster_stats = df_experiment_stats if cluster_stats is None else pd.concat([cluster_stats, df_experiment_stats], axis=1)

        cluster_mean = cluster_stats.loc['mean'].mean()
        cluster_std = cluster_stats.loc['std'].mean()

        # Calculate the difference in statistics between the cluster mean and each experiment based on the temperature
        for experiment_id, df in dict_cluster.items():
            df_experiment_stats = dict_experiment_stats.get(experiment_id)

            # Calculate the score based on the difference in statistics
            mean_diff = (df_experiment_stats['mean'] - cluster_mean) ** 2
            std_diff = (df_experiment_stats['std'] - cluster_std) ** 2
            score_mse = (mean_diff + std_diff) / 2

            cluster_experiments_scored.append((experiment_id, score_mse))

        # Sort the experiments based on the score in ascending order
        cluster_experiments_scored.sort(key=lambda x: x[1])

        # Select the top N experiments from the cluster
        dict_top_experiments[i_cluster] = {experiment_id: score_mse for experiment_id, score_mse in cluster_experiments_scored[:nr_experiments_per_cluster]}
        list_top_experiments.extend(list(dict_top_experiments[i_cluster].keys()))

    return dict_top_experiments, list_top_experiments, figure_clusters, silhouette_score_mean


if __name__ == '__main__':
    dict_dfs = {}
    files = glob.glob(f'datasets/doe_artificial_data_gan_6dpm_50smooth_10_10_S_T_F10_F10-50_F50-150_F150-300_pdp/*')
    files = glob.glob(f'datasets/doe_artificial_data_jitter_6dpm_50smooth_10_10_S_T_F10_F10-50_F50-150_F150-300_pdp/*')
    files = glob.glob(f'datasets/doe_6dpm_50smooth_1_10_S_T_F10_F10-50_F50-150_F150-300_aa/*')
    for file in files:
        file = pathlib.Path(file)
        if file.is_file() and not file.suffix:
            df = feather.read_feather(str(file))
            dict_dfs.update({file.name: df})
    synthetic_data = cluster_experiments(dict_dfs, nr_clusters=9, nr_experiments_per_cluster=1)
    a = 1
