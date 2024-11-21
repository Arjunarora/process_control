import glob
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import pandas as pd
import pyarrow.feather as feather  # Import the feather module
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Reshape, LeakyReLU, BatchNormalization, Flatten, Dropout, Conv1D, Conv1DTranspose, Conv2D, Conv2DTranspose
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.data import Dataset


GAN_LATENT_DIM = 100
GAN_NODES_G1 = 256
GAN_NODES_G2 = 512
GAN_NODES_D1 = 128
GAN_NODES_D2 = 256
GAN_BATCH_SIZE = 1
GAN_EPOCHS = 10_000
GAN_BUFFER_SIZE = 100


# https://github.com/ydataai/ydata-synthetic/blob/0.9.0/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb
def generate_artificial_data_gan(dict_dfs: dict[int, pd.DataFrame], file_model_artificial_data: pathlib.Path, num_samples=1, time_gan=None) -> [list[pd.DataFrame], TimeGAN]:

    # Extract metadata
    list_dfs = list(dict_dfs.values())
    nr_timesteps = min(len(df) for df in list_dfs)  # Trim all dataframes to the size of the smallest one
    nr_features = list_dfs[0].shape[1]
    features = list_dfs[0].columns
    index = list_dfs[0].index[:nr_timesteps]

    # Prepare data
    list_arrays_trimmed = []
    for df in list_dfs:
        i_start = (len(df) - nr_timesteps) // 2
        i_end = i_start + nr_timesteps
        array_trimmed = df[i_start:i_end].values
        array_trimmed = np.expand_dims(array_trimmed,0)
        list_arrays_trimmed.append(array_trimmed)
    array_data = np.concatenate(list_arrays_trimmed)

    # Define model parameters
    time_gan_args = ModelParameters(
        batch_size=1,
        lr=0.0001,
        noise_dim=GAN_LATENT_DIM,
        layers_dim=nr_timesteps
    )

    # If a model file was specified, load it
    if file_model_artificial_data.is_file():
        time_gan = TimeGAN.load(str(file_model_artificial_data))

    # Otherwise create, train and save a new one
    else:
        time_gan = TimeGAN(
            model_parameters=time_gan_args,
            hidden_dim=GAN_NODES_G1,
            seq_len=nr_timesteps,
            n_seq=nr_features,
            gamma=1
        )
        time_gan.train(
            array_data,
            train_steps=GAN_EPOCHS
        )
        time_gan.save(str(file_model_artificial_data))

    # Generate artificial data
    artificial_data = time_gan.sample(max(num_samples-1, 1))
    list_dfs_artificial = [pd.DataFrame(artificial_data[_], columns=features, index=index) for _ in range(num_samples)]

    return list_dfs_artificial


def generate_artificial_data_gan_selfmade(dict_dfs: dict[int, pd.DataFrame], num_samples=1) -> list[pd.DataFrame]:

    @tf.function
    def train_step(batch):

        with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
            noise = tf.random.normal([GAN_BATCH_SIZE, GAN_LATENT_DIM])

            # Generator output
            output_g = generator(noise, training=True)

            # Discriminator output for real and fake data
            output_real_d = discriminator(batch, training=True)
            output_fake_d = discriminator(output_g, training=True)

            # Generator loss
            generation_loss_g = loss_function(tf.ones_like(output_fake_d), output_fake_d)
            # deviation_loss_g = loss_function(tf.ones_like(output_g) - 0.5, output_g)
            loss_g = generation_loss_g# + deviation_loss_g

            # Discriminator loss
            real_loss_d = loss_function(tf.ones_like(output_real_d), output_real_d)
            fake_loss_d = loss_function(tf.zeros_like(output_fake_d), output_fake_d)
            loss_d = real_loss_d + fake_loss_d

        gradients_g = tape_g.gradient(loss_g, generator.trainable_variables)
        gradients_d = tape_d.gradient(loss_d, discriminator.trainable_variables)

        optimizer_g.apply_gradients(zip(gradients_g, generator.trainable_variables))
        optimizer_d.apply_gradients(zip(gradients_d, discriminator.trainable_variables))

        return loss_g, loss_d

    def train(dataset):
        for epoch in range(GAN_EPOCHS):
            time_start = time.time()

            for batch in dataset:
                loss_g, loss_d = train_step(batch)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{GAN_EPOCHS}] | {time.time() - time_start} s | D Loss: {loss_d} | G Loss: {loss_g}")

            # if (epoch + 1) % 1000 == 0:
            #     pd.DataFrame(generator(tf.random.normal([GAN_BATCH_SIZE, GAN_LATENT_DIM])).numpy()[0]).plot()
            #     plt.show()

    # Prepare data
    list_dfs = list(dict_dfs.values())
    nr_timesteps = min(len(df) for df in list_dfs)  # Trim all dataframes to the size of the smallest one
    nr_features = list_dfs[0].shape[1]
    list_arrays_trimmed = []
    for df in list_dfs:
        i_start = (len(df) - nr_timesteps) // 2
        i_end = i_start + nr_timesteps
        list_arrays_trimmed.append(df.iloc[i_start:i_end].values)

    list_arrays_trimmed = [list_arrays_trimmed[0][:, 0]]
    nr_features = 1


    # Create Generator
    # generator = Sequential()
    # generator.add(Dense(input_shape=(GAN_LATENT_DIM,), units=nr_timesteps * nr_features, use_bias=False))
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU())
    # generator.add(Reshape((nr_timesteps, nr_features)))
    # generator.add(Conv1DTranspose(filters=GAN_NODES_G1, kernel_size=5, strides=1, padding='same', use_bias=False))
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU())
    # generator.add(Conv1DTranspose(filters=GAN_NODES_G2, kernel_size=5, strides=1, padding='same', use_bias=False))
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU())
    # generator.add(Conv1DTranspose(filters=nr_features, kernel_size=5, strides=1, padding='same', use_bias=False,  activation='tanh'))

    # generator = Sequential()
    # generator.add(Dense(input_shape=(GAN_LATENT_DIM,), units=nr_timesteps * nr_features, use_bias=False))
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU())
    # generator.add(Reshape((nr_timesteps, nr_features)))
    # generator.add(Dense(units=GAN_NODES_G1, use_bias=False))
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU())
    # generator.add(Dense(units=GAN_NODES_G2, use_bias=False))
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU())
    # generator.add(Dense(units=nr_features, use_bias=False, activation='sigmoid'))

    generator = Sequential()
    generator.add(Input(shape=(GAN_LATENT_DIM,)))
    generator.add(LSTM(units=GAN_NODES_G1, return_sequences=True))
    generator.add(LSTM(units=GAN_NODES_G2, return_sequences=True))
    generator.add(Dense(units=nr_timesteps*nr_features, use_bias=False, activation='sigmoid'))
    generator.add(Reshape(target_shape=(nr_timesteps, nr_features)))

    # Create Discriminator
    # discriminator = Sequential()
    # discriminator.add(Conv1D(input_shape=(nr_timesteps, nr_features), filters=GAN_NODES_D1, kernel_size=32, strides=1, padding='same'))
    # discriminator.add(LeakyReLU())
    # discriminator.add(Dropout(0.3))
    # discriminator.add(Conv1D(filters=GAN_NODES_D2, kernel_size=32, strides=1, padding='same'))
    # discriminator.add(LeakyReLU())
    # discriminator.add(Dropout(0.3))
    # discriminator.add(Flatten())
    # discriminator.add(Dense(units=1, use_bias=False, activation='tanh'))

    # discriminator = Sequential()
    # discriminator.add(Dense(units=GAN_NODES_D1, input_shape=(nr_timesteps, nr_features), use_bias=False))
    # discriminator.add(LeakyReLU())
    # discriminator.add(Dropout(0.3))
    # discriminator.add(Dense(units=GAN_NODES_D2, use_bias=False))
    # discriminator.add(LeakyReLU())
    # discriminator.add(Dropout(0.3))
    # discriminator.add(Flatten())
    # discriminator.add(Dense(units=1, use_bias=False, activation='tanh'))

    generator = Sequential()
    generator.add(Input(shape=(nr_timesteps, nr_features)))
    generator.add(LSTM(units=GAN_NODES_D1, return_sequences=True))
    generator.add(LSTM(units=GAN_NODES_D2, return_sequences=True))
    generator.add(Dense(units=1, use_bias=False, activation='sigmoid'))

    # Define loss function
    loss_function = BinaryCrossentropy(from_logits=True)
    # loss_function = BinaryCrossentropy()
    # loss_function = MeanSquaredError()

    # Define optimizers
    optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.0002)
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.0002)

    # Actual training
    dataset = Dataset.from_tensor_slices(list_arrays_trimmed).shuffle(GAN_BUFFER_SIZE).batch(GAN_BATCH_SIZE)
    train(dataset)

    # Generate data after the final epoch
    list_dfs_artificial = []
    for i in range(num_samples):
        tensor_artificial = generator(tf.random.normal([GAN_BATCH_SIZE, GAN_LATENT_DIM]))
        df_artificial = pd.DataFrame(tensor_artificial.numpy()[0])
        list_dfs_artificial.append(df_artificial)

    df_artificial.plot()
    plt.show()

    return list_dfs_artificial


def generate_artificial_data_jitter(dict_dfs: dict[int, pd.DataFrame], num_samples=1, nr_dps_smoothing=50, jitter_level=0.05) -> list[pd.DataFrame]:

    # Initialize a list to store the synthetic dataframes
    list_dfs = list(dict_dfs.values())
    list_dfs_artificial = []

    for _ in range(num_samples):
        # Randomly select one experiment (with replacement) from the available data
        df_sample = random.choice(list_dfs).copy()

        # Apply noise
        df_artificial = df_sample + np.random.normal(0, scale=jitter_level, size=df_sample.shape)

        # Apply smoothing to the new data
        df_artificial = df_artificial.rolling(window=nr_dps_smoothing, min_periods=1).mean()

        # Append the synthetic sample to the list of synthetic dataframes
        list_dfs_artificial.append(df_artificial)

    return list_dfs_artificial


def generate_artificial_data(
        dict_dfs: dict[str: pd.DataFrame],
        art_data_method: str,
        num_samples: int,
        file_model_artificial_data: pathlib.Path,
        nr_dps_smoothing: int,
) -> [dict[str, pd.DataFrame], TimeGAN]:

    # Case statement according to the artificial data method in config.json
    if art_data_method == 'jitter':
        list_dfs_artificial = generate_artificial_data_jitter(dict_dfs, num_samples, nr_dps_smoothing, jitter_level=0.05)
    elif art_data_method == 'gan':
        list_dfs_artificial = generate_artificial_data_gan(dict_dfs, file_model_artificial_data, num_samples)
    else:
        raise NotImplementedError(f'generate_artificial_data.generate_artificial_data: Unknown data generation method "{art_data_method}".')

    return list_dfs_artificial


if __name__ == '__main__':
    list_dfs = []
    files = glob.glob(f'datasets/doe_artificial_data_gan_6dpm_50smooth_10_10_S_T_F10_F10-50_F50-150_F150-300_pdp/*')
    for i, file in enumerate(files):
        file = pathlib.Path(file)
        if file.is_file() and not file.suffix:
            df = feather.read_feather(str(file))
            list_dfs.append(df)
    dict_dfs = {i: df for i, df in enumerate(list_dfs)}
    list_dfs_artificial = generate_artificial_data_gan(dict_dfs, num_samples=1)
    # list_dfs_artificial = generate_artificial_data_jitter(dict_dfs, num_samples=5)
    file_xlsx = pathlib.Path(f'tmp/artificial_data.xlsx')
    with pd.ExcelWriter(
            file_xlsx,
            mode="a" if file_xlsx.is_file() else "w",
            date_format="YY-MM-DD hh:mm:ss.000"
    ) as writer:
        for experiment_id, df in dict_dfs.items():
            df.reset_index(drop=True).to_excel(
                writer,
                sheet_name=f'original_{experiment_id}',
            )
        for i, df in enumerate(list_dfs_artificial):
            df.reset_index(drop=True).to_excel(
                writer,
                sheet_name=f'artificial_{i}',
            )
