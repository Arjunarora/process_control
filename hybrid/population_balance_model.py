import datetime
import random
import re
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

from config import CONFIG


TF_FLOAT_DTYPE = tf.float32
TF_INT_DTYPE = tf.int32


# tf.config.run_functions_eagerly(True)  # TODO: FOR TESTING ONLY - DISABLE!
# tf.data.experimental.enable_debug_mode()  # TODO: FOR TESTING ONLY - DISABLE!


MIN_VALUE = 10 ** -100


class PopulationBalanceModel:
    """
    Variables and indices from [Gunawan 2004] if not otherwise specified
    t                                                       [s] time
    l                                                       [µm] size
    k                                                       [s] time interval
    h                                                       [µm] length interval
    m                                                       [#] time index
    n                                                       [#] length index
    f(l, t)                                                 [#/(m^3)] christal size distribution - number of crystals of size l at time t
    j = k1 * s * e ^ (-k2 / (ln(s) ** 2))                    [#/(m^3s)] nucleation rate from [Vetter 2013]
    g = k1 * (c - c_sat) ** k2                               [m/s] size independent growht from [Vetter 2013]
    g = k1 * ((c - c_sat) / c_sat) ** k2 * (1 + 0.1 * l)    [m/s] size dependent growth

    Equations from [Vetter 2013] if not otherwise specified
    df/dt = b - d - d(g f)/dl                               population Balance Equation
    dc/dt = -dm/dt                                          mass Balance
    b = j * dirac(l - l_crit)                               birth rate: Only spontaneous nucleation is regarded
    d = const.                                              assumed constant across all sizes

    The pbe model can be made discrete [Marchal 1988]
    d(g f)/dl becomes [g(l_i) * f(t, l_i) - g(l_i-1) * f(t, l_i-1)]

    Starting conditions from [Vetter 2013]
    f(t=0, l) = f0(l)                                       starting CSD - 0 for all l if no seed crystals are used
    f(t, l=0) = j / g                                       spontaneous nucleation
    c(t=0) = c0                                             starting concentration

    Units are from [Hilfiker 2018: Crystallization Process Modelling]
    """

    # @tf.function
    def __init__(self, **kwargs):

        # Features
        self.features_history = kwargs.get("features_history")
        self.features_controlled = kwargs.get("features_controlled")
        self.features_prediction = kwargs.get("features_prediction")
        self.features_hybrid = kwargs.get("features_hybrid")
        self.nr_dps_per_minute = kwargs.get("nr_dps_per_minute")
        self.nr_timesteps_history = kwargs.get("nr_timesteps_history")
        self.nr_timesteps_prediction = kwargs.get("nr_timesteps_prediction")

        # Size l, size index n, size interval h
        h = kwargs.get("h", 0.1)  # [µm] size interval
        self.h = tf.convert_to_tensor(h, dtype=TF_FLOAT_DTYPE)
        l_max = kwargs.get("l_max", 300)  # [µm] Range of size that will be calculated
        self.l_max = tf.convert_to_tensor(l_max, dtype=TF_FLOAT_DTYPE)
        n_max = int(l_max / h)  # [#] max index for size intervals
        self.n_max_py = n_max
        self.n_max = tf.convert_to_tensor(n_max, dtype=TF_INT_DTYPE)
        self.columns_l = tf.linspace(0.0, l_max, n_max)

        # Time t, time index m, time interval k
        k = kwargs.get("k", 0.1)  # [s] time interval
        self.k = tf.convert_to_tensor(k, dtype=TF_FLOAT_DTYPE)
        t_max = self.nr_timesteps_prediction / self.nr_dps_per_minute * 60  # [s] Time range that will be calculated
        self.t_max = tf.convert_to_tensor(t_max, dtype=TF_FLOAT_DTYPE)
        m_max = int(t_max / k)  # [#] max index for time intervals
        self.m_max_py = m_max
        self.m_max = tf.convert_to_tensor(m_max, dtype=TF_INT_DTYPE)
        self.columns_t = tf.linspace(0.0, t_max, m_max)

        # Set time interval based on Courant-Friedrichs-Levy (CFL) condition
        # | g_max * k/h | <= 1  # g_max, k and h assumed to always be positive
        # k <= h / g_max
        # if not kwargs.get("k"):
        #     self.g_max = self.get_size_dependent_growth_rate(self.n_max, self.c_0)
        #     self.k = self.h / self.g_max

        # Substance data
        self.substance = kwargs.get("substance")  # "aa", "pdp"
        self.density = kwargs.get("density")  # 1.3, 2.34
        self.c_dissolved = tf.convert_to_tensor(50 if self.substance == "aa" else 335, dtype=TF_FLOAT_DTYPE)  # [g/L] Concentration if all crystals are dissolved  # todo: Dissolved concentration from config?
        self.l_critical = tf.convert_to_tensor(kwargs.get("l_critical", 5.5), dtype=TF_FLOAT_DTYPE)  # µm - Critical size of stable crystals
        self.n_critical = self.size_to_index(self.l_critical)

        # Process data
        self.volume_reactor = 2  # L

        # Init lookup dicts
        self.dict_features_history = {feature: i for i, feature in enumerate(self.features_history)}
        self.dict_features_controlled = {feature: i for i, feature in enumerate(self.features_controlled)}
        self.dict_features_hybrid = {feature: i for i, feature in enumerate(self.features_hybrid)}

    @tf.function
    def index_to_time(self, m: tf.Tensor) -> tf.Tensor:
        return tf.cast(m * self.k, dtype=TF_FLOAT_DTYPE)

    @tf.function
    def time_to_index(self, t: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.math.round(t / self.k), dtype=TF_INT_DTYPE)

    @tf.function
    def index_to_size(self, n: tf.Tensor) -> tf.Tensor:
        return tf.cast(n * self.h, dtype=TF_FLOAT_DTYPE)

    @tf.function
    def size_to_index(self, l: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.math.round(l / self.h), dtype=TF_INT_DTYPE)

    @staticmethod
    @tf.function
    def get_flux_limiter(f_m_n_minus1: tf.Tensor, f_m_n: tf.Tensor, f_m_n_plus1: tf.Tensor) -> tf.Tensor:
        """
        Calculates simple flux limiter. Can handle Scalars and 1-dim Tensors.
        """
        theta = tf.where(
            f_m_n_plus1 - f_m_n == 0,
            0.0,
            (f_m_n - f_m_n_minus1) / (f_m_n_plus1 - f_m_n)
        )
        flux_limiter = (tf.abs(theta) + theta) / (1 + tf.abs(theta))
        return flux_limiter

    @tf.function
    def get_surface_total(self, f_m: tf.Tensor) -> tf.Tensor:
        surfaces = f_m * 4 * tf.math.pi * self.index_to_size(self.columns_l * tf.cast(self.h, TF_FLOAT_DTYPE) / 2) ** 2 / 1000 ** 6
        surface = tf.reduce_sum(surfaces, axis=-1)
        # surface /= 1000 ** 6
        return surface

    @tf.function
    def get_volume_total(self, f_m: tf.Tensor) -> tf.Tensor:
        vs = f_m * 5.235987755982988e-19 * self.index_to_size(self.columns_l * tf.cast(self.h, TF_FLOAT_DTYPE)) ** 3
        volume = tf.reduce_sum(vs, axis=-1)
        # volume /= 1000 ** 6
        return volume

    @tf.function
    def get_mass_total(self, f_m: tf.Tensor) -> tf.Tensor:
        return self.get_volume_total(f_m) * self.density

    @tf.function
    def get_f_from_count_bins(self, count_bins: tf.Tensor, bin_names: list[str], mass_total: tf.Tensor) -> tf.Tensor:
        """
        Generates an f distribution from FBRM count bins, assuming a normal distribution.
        FBRM counts are assumed to be absolute counts per 2m laser driveway.
        """
        # Get statistics from bins
        counts_sum = tf.reduce_sum(count_bins, axis=-1)  # Total absolute counts in all bins together
        counts_sum = tf.expand_dims(counts_sum, -1)  # Reshape for batch compatibility
        h_centers = tf.convert_to_tensor([(self.get_count_bin_limits(bin_name)[0] + self.get_count_bin_limits(bin_name)[1]) / 2 for bin_name in bin_names], dtype=TF_FLOAT_DTYPE)
        l_mean = tf.reduce_sum(count_bins / counts_sum * h_centers, axis=-1)
        l_std = tf.math.reduce_std(count_bins / counts_sum * h_centers, axis=1)

        # Fit normal distribution to count bin percentages per 2m
        norm_dist = tfp.distributions.Normal(loc=l_mean, scale=l_std)
        f_2m = tf.TensorArray(TF_FLOAT_DTYPE, size=len(self.columns_l), dynamic_size=False, clear_after_read=False)
        for i, l in enumerate(self.columns_l):
            f_2m = f_2m.write(i, norm_dist.prob(l))
        f_2m = tf.transpose(f_2m.stack())

        # Convert f per 2m to percentage based f
        f_2m_sum = tf.reduce_sum(f_2m, axis=-1)
        f_2m_sum = tf.expand_dims(f_2m_sum, -1)  # Reshape for batch compatibility
        f = f_2m / f_2m_sum

        # Convert percentage based f to absolute f
        mass_f = tf.expand_dims(self.get_mass_total(f), -1)
        mass_total = tf.expand_dims(mass_total, -1)
        f = f * mass_total / mass_f

        return f

    @tf.function
    def get_count_bins_from_f(self, f_m: tf.Tensor, bin_names: list[str], percentage=True) -> tf.Tensor:
        """
        Generates FBRM count bins from f. The bins will contain the absolute values of the whole system, which will
        probably be higher than the measurement counts only registering crystals in the measurement volume.
        Can be converted to percentage based count bins with the percentage argument.
        """
        count_bins_raw = tf.TensorArray(TF_FLOAT_DTYPE, size=len(bin_names), dynamic_size=False, clear_after_read=False)
        size_limits = [self.get_count_bin_limits(bin_name) for bin_name in bin_names]
        index_limits = [[
            tf.cast(self.size_to_index(size_limit[0]), TF_FLOAT_DTYPE),
            tf.cast(self.size_to_index(size_limit[1])-1, TF_FLOAT_DTYPE)
        ] for size_limit in size_limits]

        for i, bin_name in enumerate(bin_names):
            mask = tf.where(
                tf.logical_and(tf.math.greater_equal(self.columns_l, index_limits[i][0]), tf.math.less(self.columns_l, index_limits[i][1])),
                f_m,
                tf.zeros(shape=tf.shape(f_m), dtype=TF_FLOAT_DTYPE)
            )
            counts = tf.reduce_sum(mask, axis=-1)
            count_bins_raw = count_bins_raw.write(i, counts)
        count_bins_raw = tf.transpose(count_bins_raw.stack(), [1, 2, 0])

        # Convert to percentage of total counts
        if percentage:  # Breaks GradientTape
            sum_counts = tf.math.reduce_sum(f_m, axis=-1)
            sum_counts = tf.expand_dims(sum_counts, -1)
            count_bins = tf.where(
                tf.less_equal(sum_counts, 0.0),
                count_bins_raw * 0.0,
                count_bins_raw / sum_counts
            )
        else:
            count_bins = count_bins_raw
        return count_bins

    @tf.function
    def get_f_m_plus1(self, f_m: tf.Tensor, c_m: tf.Tensor, temperature_m: tf.Tensor, g_k1: tf.Tensor, g_k2: tf.Tensor, solver="h1") -> tf.Tensor:

        # Shift f to create tensors representing other n values
        # todo: Set edge values to 0
        f_m_n_minus2 = tf.roll(f_m, shift=-2, axis=-1)
        f_m_n_minus1 = tf.roll(f_m, shift=-1, axis=-1)
        f_m_n_plus1 = tf.roll(f_m, shift=1, axis=-1)

        flux_limiter_n = self.get_flux_limiter(f_m_n_minus1, f_m, f_m_n_plus1)
        flux_limiter_n_minus1 = self.get_flux_limiter(f_m_n_minus2, f_m_n_minus1, f_m)

        # HR1
        if solver == "h1":
            g_n = self.get_size_dependent_growth_rate(self.columns_l, c_m, temperature_m, g_k1, g_k2, self.substance)
            g_n_minus1 = tf.roll(g_n, shift=-1, axis=0)
            f_m_plus1 = f_m - self.k / self.h * (g_n * f_m - g_n_minus1 * f_m_n_minus1) - (
                (self.k * g_n / (2 * self.h)) * (1 - self.k * g_n / self.h) * (f_m_n_plus1 - f_m) * flux_limiter_n -
                (self.k * g_n_minus1 / (2 * self.h)) * (1 - self.k * g_n_minus1 / self.h) * (f_m - f_m_n_minus1) * flux_limiter_n_minus1
            )
        else:
            raise NotImplementedError()

        return f_m_plus1

    @tf.function
    def call(self, input_history: tf.Tensor, input_controlled: tf.Tensor, output_ann: tf.Tensor) -> tf.Tensor:
        """
        Method that will be called from NN layer to get an output for a future timestep.
        It will calculate future timesteps in the prediction range.
        """
        t0 = datetime.datetime.now()
        print("TRACING PopulationBalanceModel.call")

        batch_size = tf.shape(input_history)[0]

        # Init TensorArrays to store data
        f = tf.TensorArray(TF_FLOAT_DTYPE, size=self.m_max_py, dynamic_size=False, clear_after_read=False)

        # Set starting conditions
        c_m_normalized = tf.math.reduce_mean(input_history[:, :, self.dict_features_history.get("ftir:concentration")], axis=1)
        c_m = self.denormalize_tensor(c_m_normalized, "ftir:concentration")

        mass_total_m = (self.c_dissolved - c_m) * self.volume_reactor
        mass_total_m = tf.nn.relu(mass_total_m)  # Set negative values to zero

        cbs_normalized = tf.gather(input_history[:, :], [self.dict_features_history.get(feature) for feature in self.features_prediction], axis=2)
        count_bins_m_normalized = tf.math.reduce_mean(cbs_normalized, axis=1)
        count_bins_m = self.denormalize_tensor(count_bins_m_normalized, "fbrm:counts_10")  # todo: denormalize for each FBRM feature specifically

        f_m = self.get_f_from_count_bins(count_bins_m, self.features_prediction, mass_total_m)
        f_m = tf.nn.relu(f_m)  # Set negative values to zero

        j_k1 = output_ann[:, self.dict_features_hybrid.get("pbe:j_k1")] * 10.0 ** -5  # 5 * 10 ** -7
        j_k2 = output_ann[:, self.dict_features_hybrid.get("pbe:j_k2")] * 10.0 ** 1  # 1.7
        g_k1 = output_ann[:, self.dict_features_hybrid.get("pbe:g_k1")] * 10.0 ** 1  # 1
        g_k2 = output_ann[:, self.dict_features_hybrid.get("pbe:g_k2")] * 10.0 ** 1  # 1.3

        # Create temperature array in PBE model time step size from future_temperatures
        future_temperatures = tf.convert_to_tensor(input_controlled[:, :, self.dict_features_controlled.get("thermostat:temperature_setpoint")], dtype=TF_FLOAT_DTYPE)
        future_temperatures = self.denormalize_tensor(future_temperatures, "thermostat:temperature_setpoint")
        future_temperatures_pbe = tf.linspace(future_temperatures[:, 0], future_temperatures[:, -1], self.m_max_py)

        # Update f in prediction range
        for m, temperature_m_plus_1 in enumerate(future_temperatures_pbe[1:-1]):

            # Growth
            f_m_plus1_growth = self.get_f_m_plus1(f_m, c_m, temperature_m_plus_1, g_k1, g_k2)  # Shape (500,)

            # Spontaneous nucleation
            j = self.get_nucleation_rate(c_m, temperature_m_plus_1, j_k1, j_k2, self.substance)
            f_m_plus1_nucleation = tf.transpose(tf.scatter_nd(
                indices=[[self.n_critical]],
                updates=[j],
                shape=(self.n_max, batch_size)
            ))

            # Update distribution
            f_m_plus1 = f_m_plus1_growth + f_m_plus1_nucleation
            f_m_plus1 = tf.nn.relu(f_m_plus1)  # Set negative values to zero

            # Update concentration based on mass balance and changes in f
            c_m_plus1 = self.get_mass_total(f_m_plus1) / self.volume_reactor
            c_m_plus1 = tf.nn.relu(c_m_plus1)  # Set negative values to zero

            # Update current data
            f = f.write(m, f_m)
            f_m = f_m_plus1
            c_m = c_m_plus1

        # Get prediction
        f = tf.transpose(f.stack(), [1, 0, 2])
        steps = int(self.m_max_py / self.nr_timesteps_prediction)
        f_prediction = f[:, ::steps, :]
        prediction = self.get_count_bins_from_f(f_prediction, self.features_prediction)

        t1 = datetime.datetime.now()
        print(f"TRACING PopulationBalanceModel.call took {(t1-t0).total_seconds()}s")
        tf.print(f"PopulationBalanceModel.call took {(t1-t0).total_seconds()}s")
        return prediction

    @staticmethod
    def plot(f_m: tf.Tensor):
        plt.plot(f_m)
        plt.show()

    def plot_3d(self, f):

        # Load and format data
        z = f
        nrows, ncols = z.shape
        x = np.linspace(0, self.n_max, ncols)
        y = np.linspace(0, ncols, nrows)
        x, y = np.meshgrid(x, y)

        # Set up plot
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ls = LightSource(270, 45)
        # To use a custom hillshading mode, override the built-in shading and pass
        # in the rgb colors of the shaded surface calculated from "shade".
        # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        rgb = None
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)
        plt.show()

    @staticmethod
    @tf.function
    def denormalize_tensor(t: tf.Tensor, feature: str) -> tf.Tensor:
        normalization_range = CONFIG.get("dataset", {}).get("normalization_ranges", {}).get(feature)
        t_denormalized = t * (normalization_range[1] - normalization_range[0]) + normalization_range[0]
        return t_denormalized

    @staticmethod
    @tf.function
    def get_saturation_concentration(temperature: tf.Tensor, substance: str) -> tf.Tensor:
        """
        Calculates the saturation concentration of a substance at given temperature in water.
        :param temperature: System temperature in °C
        :param substance: Substance name in lowercase, e.g. "adipic_acid" or "kh2po4"
        :return: The saturation concentration of a given substance in water
        """
        if substance in ["adipic_acid", "aa", "adipin"]:
            c_sat = 13.505 * tf.math.exp(0.0418 * temperature)
        elif substance in ["potassium_dihydrogen_phosphate", "kh2po4", "pdp", "kdp"]:
            temperature += 273.15
            c_sat = (4.6479 * 10 ** -5 * temperature ** 2 - 0.022596 * temperature + 2.8535) * 1000
        else:
            raise NotImplementedError(
                f"feature_engineering.get_saturation_concentration: Substance {substance} is not implemented yet.")
        return c_sat

    @tf.function
    def get_supersaturation(self, concentration: tf.Tensor, temperature: tf.Tensor, substance: str) -> tf.Tensor:
        """
        Calculates the oversaturation of a substance in water at given temperature
        :param concentration: Concentration of the substance in the system in g/kg_H2O
        :param temperature: System temperature in °C
        :param substance: Substance name in lowercase, e.g. "adipic_acid" or "kh2po4"
        :return:
        """
        c_sat = self.get_saturation_concentration(temperature, substance)
        s = concentration / c_sat
        return s

    @tf.function
    def get_nucleation_rate(self, c: tf.Tensor, temperature: tf.Tensor, k1: tf.Tensor, k2: tf.Tensor, substance: str) -> tf.Tensor:
        """
        From [Vetter 2013]
        Calculates the nucleation rate in #/(m^3s^)
        :param c: concentration
        :param temperature:
        :param k1:
        :param k2:
        :param substance:
        :return:
        """
        s = self.get_supersaturation(c, temperature, substance)
        j = k1 * s * tf.math.exp(-k2 / (tf.math.log(s) ** 2))
        return j

    @tf.function
    def get_size_independent_growth_rate(self, c: tf.Tensor, temperature: tf.Tensor, k1: tf.Tensor, k2: tf.Tensor, substance: str) -> tf.Tensor:
        """
        From [Vetter 2013]
        Calculates the size independant growth rate
        :param c: concentration
        :param temperature: temperature
        :param k1:
        :param k2:
        :param substance:
        :return:
        """
        c_sat = self.get_saturation_concentration(temperature, substance)
        g = k1 * (c - c_sat) ** k2
        return g

    @tf.function
    def get_size_dependent_growth_rate(self, l: tf.Tensor, c: tf.Tensor, temperature: tf.Tensor, k1: tf.Tensor, k2: tf.Tensor, substance: str) -> tf.Tensor:
        """
        From [Gunawan 2004]
        Calculates the size dependant growth rate
        :param l: size
        :param c: concentration
        :param temperature: temperature
        :param k1:
        :param k2:
        :return:
        """
        c_sat = self.get_saturation_concentration(temperature, substance)
        s = (c - c_sat) / c_sat
        s = tf.nn.relu(s)  # Set negative values to zero
        g = tf.expand_dims(k1 * s ** k2, -1) * (1 + 0.1 * tf.convert_to_tensor(l, dtype=TF_FLOAT_DTYPE))
        return g

    @staticmethod
    @tf.function
    def get_test_process_temperature(temperature_0: tf.Tensor, temperature_M: tf.Tensor, t: tf.Tensor, t_max=tf.convert_to_tensor(100.0, dtype=TF_FLOAT_DTYPE)):
        return temperature_0 - t/t_max * (temperature_0 - temperature_M)

    @staticmethod
    @tf.function
    def get_count_bin_limits(bin_name: str) -> tf.Tensor:
        """
        Parses the names of FBRM count bins and returns their size limits
        :param bin_name:
        :return:
        """
        matches = re.findall(r"\d+-\d+$", bin_name)
        if matches:
            limits = tuple([int(_) for _ in matches[0].split("-")])
        elif bin_name.endswith("_10"):
            limits = (0, 10)
        elif bin_name.endswith("_1000"):
            limits = (1000, 1001)
        else:
            limits = (0, 0)
        return tf.convert_to_tensor(limits, dtype=TF_FLOAT_DTYPE)


@tf.keras.utils.register_keras_serializable()
class ModelPBE(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs_pbe = None
        self.pbe = None

    def init_pbe(self, kwargs_pbe):
        self.kwargs_pbe = kwargs_pbe
        self.pbe = PopulationBalanceModel(**self.kwargs_pbe)

    @tf.function
    def train_step(self, data):
        t0 = datetime.datetime.now()
        print("TRACING ModelPBE.train_step")

        # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
        x, y = data
        input_history = x.get("history")
        input_controlled = x.get("controlled")

        # Convert y_true labels to percentage based FBRM counts, since the y_pred predictions are percentages
        y_true_sum = tf.reduce_sum(y, axis=-1)  # Total counts per timestep in shape (20, 10)
        y_true_sum = tf.expand_dims(y_true_sum, axis=-1)  # Total counts per timestep in shape (20, 10, 1)
        y_true = y / y_true_sum

        with tf.GradientTape() as tape:
            output_ann = self(x, training=True)  # Forward pass

            # Compute the loss value (the loss function is configured in `compile()`)
            # Percentages of counts per FBRM bin in shape (20, 10, 4)
            y_pred = self.pbe.call(
                input_history=input_history,
                input_controlled=input_controlled,
                output_ann=output_ann
            )
            loss = self.compute_loss(y=y_true, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables

        # todo: tape.gradient returns nan, even if loss and trainable_vars exist on the first iteration.
        #  After apply_gradients, the weights of the model are also nan, so that from there on loss and weights are always nan.
        #  https://www.tensorflow.org/guide/autodiff
        gradients = tape.gradient(loss, trainable_vars)  # This takes the longest time within this method

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        t1 = datetime.datetime.now()
        print(f"TRACING ModelPBE.train_step took {(t1-t0).total_seconds()}s")
        tf.print(f"ModelPBE.train_step took {(t1-t0).total_seconds()}s")
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update({
            "kwargs_pbe": self.kwargs_pbe
        })
        return config

    @classmethod
    def from_config(cls, config):
        kwargs_pbe = config.pop("kwargs_pbe")
        instance = super().from_config(config)
        instance.init_pbe(kwargs_pbe)
        return instance


if __name__ == '__main__':
    import random

    features_history = CONFIG.get("features", {}).get("features_history")
    features_controlled = CONFIG.get("features", {}).get("features_controlled")
    features_prediction = CONFIG.get("features", {}).get("features_prediction")
    features_hybrid = CONFIG.get("features", {}).get("features_hybrid")
    nr_dps_per_minute = CONFIG.get("dataset", {}).get("nr_dps_per_minute")
    nr_timesteps_history = CONFIG.get("model", {}).get("nr_timesteps_history")
    nr_timesteps_prediction = CONFIG.get("model", {}).get("nr_timesteps_prediction")
    batch_size = CONFIG.get("model", {}).get("batch_size")
    substance = CONFIG.get("dataset", {}).get("substance")
    density = CONFIG.get("dataset", {}).get("density")

    def get_test_inputs():
        input_history = [
            [
                [
                    0.0 if feature == "thermostat:temperature_process"
                    else random.uniform(0.75, 0.75) if feature == "ftir:concentration"
                    else random.uniform(0.05, 0.1) if feature == "fbrm:counts_10"
                    else random.uniform(0.2, 0.4) if feature == "fbrm:counts_10-50"
                    else random.uniform(0.5, 0.7) if feature == "fbrm:counts_50-150"
                    else random.uniform(0.2, 0.4) if feature == "fbrm:counts_150-300"
                    else random.uniform(0, 1) for feature in features_history
                ] for timestep in range(nr_timesteps_history)
            ] for batch in range(batch_size)
        ]
        input_controlled = [[[(1 - timestep/nr_timesteps_prediction) * 0.6 for feature in features_controlled] for timestep in range(nr_timesteps_prediction)] for batch in range(batch_size)]
        output_ann = [[random.uniform(0, 1) for feature in features_hybrid] for batch in range(batch_size)]
        inputs = [
            tf.convert_to_tensor(input_history),
            tf.convert_to_tensor(input_controlled),
            tf.convert_to_tensor(output_ann),
        ]
        return inputs

    t0 = datetime.datetime.now()
    pbe = PopulationBalanceModel(
        features_history=features_history,
        features_controlled=features_controlled,
        features_prediction=features_prediction,
        features_hybrid=features_hybrid,
        nr_dps_per_minute=nr_dps_per_minute,
        nr_timesteps_history=nr_timesteps_history,
        nr_timesteps_prediction=nr_timesteps_prediction,
        h=0.1,
        l_max=300,
        l_critical=5.5,
        k=0.1,
        substance=substance,
        density=density
    )
    t1 = datetime.datetime.now()
    prediction1 = pbe.call(get_test_inputs())
    t2 = datetime.datetime.now()
    prediction2 = pbe.call(get_test_inputs())
    t3 = datetime.datetime.now()
    prediction3 = pbe.call(get_test_inputs())
    t4 = datetime.datetime.now()
    prediction4 = pbe.call(get_test_inputs())
    t5 = datetime.datetime.now()
    print(f'creation: {(t1-t0).total_seconds()}s, prediction1: {(t2-t1).total_seconds()}s, prediction2: {(t3-t2).total_seconds()}s, prediction3: {(t4-t3).total_seconds()}s, prediction4: {(t5-t4).total_seconds()}s')
    # creation: 0.153718s, prediction1: 80.919165s, prediction2: 1.654779s, prediction3: 1.624568s, prediction4: 1.651701s
    a=1
