import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Layer, Input, Dense, LSTM, Concatenate, Reshape, GRU
import mlflow
import pathlib

from config import CONFIG, RELEVANT_SECTIONS
from hybrid.population_balance_model import ModelPBE


def create_model(config=CONFIG):
    # Import common configuration parameters
    nr_features_history = config.get('features', {}).get('nr_features_history')
    nr_features_prediction = config.get('features', {}).get('nr_features_prediction')
    nr_features_controlled = config.get('features', {}).get('nr_features_controlled')

    model_name = config.get('model', {}).get('model_name')
    model_type = config.get('model', {}).get('model_type', 'lstm')
    nr_timesteps_history = config.get('model', {}).get('nr_timesteps_history')
    nr_timesteps_prediction = config.get('model', {}).get('nr_timesteps_prediction')
    batch_size = config.get('model', {}).get('batch_size')

    file_model_config = config.get('paths', {}).get('file_model_config')
    file_model_predict_config = config.get('paths', {}).get('file_model_predict_config')

    # Catch invalid parameters and cast types if necessary
    if not isinstance(nr_features_history, int):
        raise TypeError(f'Expected type int for nr_features_history, got: {type(nr_features_history)}.')
    if nr_features_history < 1:
        raise ValueError(f'Invalid value {nr_features_history} for nr_features_history given.')

    if not isinstance(nr_features_prediction, int):
        raise TypeError(f'Expected type int for nr_features_prediction, got: {type(nr_features_prediction)}.')
    if nr_features_prediction < 1:
        raise ValueError(f'Invalid value {nr_features_prediction} for nr_features_prediction given.')

    if not isinstance(nr_features_controlled, int):
        raise TypeError(f'Expected type int for nr_features_controlled, got: {type(nr_features_controlled)}.')
    if nr_features_controlled < 1:
        raise ValueError(f'Invalid value {nr_features_controlled} for nr_features_controlled given.')

    if not isinstance(model_name, str):
        raise TypeError(f'Expected type str for model_name, got: {type(model_name)}.')

    if isinstance(model_type, list):
        model_type = model_type[0]
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

    if isinstance(batch_size, list):
        batch_size = batch_size[0]
    if not isinstance(batch_size, int):
        raise TypeError(f'Expected type int for batch_size, got: {type(batch_size)}.')
    if batch_size < 1:
        raise ValueError(f'Invalid value {batch_size} for batch_size given.')

    if not isinstance(file_model_config, pathlib.Path):
        raise TypeError(f'Expected type pathlib.Path for file_model_config, got: {type(file_model_config)}.')

    if not isinstance(file_model_predict_config, pathlib.Path):
        raise TypeError(
            f'Expected type pathlib.Path for file_model_predict_config, got: {type(file_model_predict_config)}.')

    # Log relevant config params
    for section in RELEVANT_SECTIONS.get('create_model'):
        for key, val in config.get(section, {}).items():
            if len(str(val)) >= 250:
                continue
            mlflow.log_param(key, val)

    # Assemble dict with model specific parameters
    model_kwargs = config.get("dataset", {})
    model_kwargs.update(config.get("features", {}))
    model_kwargs.update(config.get("model", {}))
    model_kwargs_prediction = model_kwargs.copy()
    model_kwargs_prediction.update({"batch_size": 1})

    # Select model creation function
    if model_type == 'ffn':
        func_model_creation = _create_model_ffn
    elif model_type == 'lstm':
        func_model_creation = _create_model_lstm
    elif model_type == 'gru':
        func_model_creation = _create_model_gru
    # elif model_type == 'hybrid_ffn':
    #     func_model_creation = _create_model_hybrid_ffn
    elif model_type in ['hybrid_lstm', 'lstm_hybrid', 'hybrid']:
        func_model_creation = _create_model_hybrid_lstm
    # elif model_type == 'hybrid_gru':
    #     func_model_creation = _create_model_hybrid_gru
    else:
        raise ValueError(f'create_model.create_model: Invalid model_type {model_type}.')

    # Create model for training
    model = func_model_creation(**model_kwargs)

    # Create model for prediction (different batch size = 1)
    model_predict = func_model_creation(**model_kwargs_prediction)

    # Save trainable model
    with open(file_model_config, "w") as w:
        w.write(model.to_json())
    mlflow.log_artifact(str(file_model_config))

    # Save model for prediction
    with open(file_model_predict_config, "w") as w:
        w.write(model_predict.to_json())
    mlflow.log_artifact(str(file_model_predict_config))


def _create_model_ffn(**kwargs):
    model_name = kwargs.get("model_name")
    list_units_per_layer = kwargs.get("list_units_per_layer")
    list_dropout_per_layer = kwargs.get("list_dropout_per_layer")
    kernel_regulizer = kwargs.get("kernel_regulizer")
    bias_regulizer = kwargs.get("bias_regulizer")
    dense_activation_function = kwargs.get("dense_activation_function")
    batch_size = kwargs.get("batch_size")
    nr_timesteps_history = kwargs.get("nr_timesteps_history")
    nr_timesteps_prediction = kwargs.get("nr_timesteps_prediction")
    nr_features_history = kwargs.get("nr_features_history")
    nr_features_controlled = kwargs.get("nr_features_controlled")
    nr_features_prediction = kwargs.get("nr_features_prediction")

    if not isinstance(list_units_per_layer, list) or any(_ for _ in list_units_per_layer if not isinstance(_, int)):
        raise TypeError(f'Expected type list[int] for list_units_per_layer, got: {type(list_units_per_layer)}.')
    if len(list_units_per_layer) < 1:
        raise ValueError(f'Invalid value {list_units_per_layer} for list_units_per_layer given.')

    if not isinstance(list_dropout_per_layer, list) or any(_ for _ in list_dropout_per_layer if not isinstance(_, int)):
        raise TypeError(f'Expected type list[int] for list_dropout_per_layer, got: {type(list_dropout_per_layer)}.')
    if len(list_dropout_per_layer) != len(list_units_per_layer):
        raise ValueError(f'Invalid value {list_dropout_per_layer} for list_dropout_per_layer given.')

    if isinstance(kernel_regulizer, list):
        kernel_regulizer = kernel_regulizer[0]
    if not isinstance(kernel_regulizer, str):
        raise TypeError(f'Expected type str for kernel_regulizer, got: {type(kernel_regulizer)}.')

    if isinstance(bias_regulizer, list):
        bias_regulizer = bias_regulizer[0]
    if not isinstance(bias_regulizer, str):
        raise TypeError(f'Expected type str for bias_regulizer, got: {type(bias_regulizer)}.')

    if isinstance(dense_activation_function, list):
        dense_activation_function = dense_activation_function[0]
    if not isinstance(dense_activation_function, str):
        raise TypeError(f'Expected type str for dense_activation_function, got: {type(dense_activation_function)}.')

    # Input history
    layer_input_history = Input(
        batch_input_shape=(batch_size, nr_timesteps_history, nr_features_history),
        name='history'
    )
    layer_input_history_reshaped = Reshape(
        target_shape=(nr_timesteps_history * nr_features_history,)
    )(layer_input_history)

    layer_ffn_history = None
    for i, (units, dropout) in enumerate(zip(list_units_per_layer, list_dropout_per_layer)):
        layer_ffn_history = Dense(
            units,
            activation=dense_activation_function,
            batch_input_shape=(1 if i == 0 else None, nr_timesteps_history, nr_features_history),
            use_bias=True,
            kernel_regularizer=kernel_regulizer,
            bias_regularizer=bias_regulizer
        )(layer_input_history_reshaped if i == 0 else layer_ffn_history)

    # Input controlled
    layer_input_controlled = Input(
        batch_input_shape=(batch_size, nr_timesteps_prediction, nr_features_controlled),
        name='controlled')
    layer_input_controlled_reshaped = Reshape(
        target_shape=(nr_timesteps_prediction * nr_features_controlled,)
    )(layer_input_controlled)

    layer_ffn_controlled = Dense(
        list_units_per_layer[-1],
        activation=dense_activation_function,
        batch_input_shape=(batch_size, nr_timesteps_prediction, nr_features_controlled),
        use_bias=True,
        kernel_regularizer=kernel_regulizer,
        bias_regularizer=bias_regulizer
    )(layer_input_controlled_reshaped)

    layer_combined = Concatenate(
        axis=1
    )([layer_ffn_history, layer_ffn_controlled])

    # Output layer
    layer_output = Dense(
        nr_timesteps_prediction * nr_features_prediction,
        activation=dense_activation_function
    )(layer_combined)

    # Reshape for correct nr of output features if multiple are to be predicted
    if nr_features_prediction > 1:
        layer_output = Reshape(
            target_shape=(nr_timesteps_prediction, nr_features_prediction)
        )(layer_output)  # Output Layer

    # Assemble model
    model = Model(
        [layer_input_history, layer_input_controlled],
        layer_output,
        name=model_name
    )

    return model


def _create_model_lstm(**kwargs):
    model_name = kwargs.get("model_name")
    list_units_per_layer = kwargs.get("list_units_per_layer")
    list_dropout_per_layer = kwargs.get("list_dropout_per_layer")
    stateful = kwargs.get("stateful")
    kernel_regulizer = kwargs.get("kernel_regulizer")
    bias_regulizer = kwargs.get("bias_regulizer")
    dense_activation_function = kwargs.get("dense_activation_function")
    batch_size = kwargs.get("batch_size")
    nr_timesteps_history = kwargs.get("nr_timesteps_history")
    nr_timesteps_prediction = kwargs.get("nr_timesteps_prediction")
    nr_features_history = kwargs.get("nr_features_history")
    nr_features_controlled = kwargs.get("nr_features_controlled")
    nr_features_prediction = kwargs.get("nr_features_prediction")

    if not isinstance(list_units_per_layer, list) or any(_ for _ in list_units_per_layer if not isinstance(_, int)):
        raise TypeError(f'Expected type list[int] for list_units_per_layer, got: {type(list_units_per_layer)}.')
    if len(list_units_per_layer) < 1:
        raise ValueError(f'Invalid value {list_units_per_layer} for list_units_per_layer given.')

    if not isinstance(list_dropout_per_layer, list) or any(_ for _ in list_dropout_per_layer if not isinstance(_, int)):
        raise TypeError(f'Expected type list[int] for list_dropout_per_layer, got: {type(list_dropout_per_layer)}.')
    if len(list_dropout_per_layer) != len(list_units_per_layer):
        list_dropout_per_layer = [list_dropout_per_layer[0] for _ in range(len(list_units_per_layer))]

    if not isinstance(stateful, bool):
        raise TypeError(f'Expected type bool for stateful, got: {type(stateful)}.')

    if isinstance(kernel_regulizer, list):
        kernel_regulizer = kernel_regulizer[0]
    if not isinstance(kernel_regulizer, str):
        raise TypeError(f'Expected type str for kernel_regulizer, got: {type(kernel_regulizer)}.')

    if isinstance(bias_regulizer, list):
        bias_regulizer = bias_regulizer[0]
    if not isinstance(bias_regulizer, str):
        raise TypeError(f'Expected type str for bias_regulizer, got: {type(bias_regulizer)}.')

    if isinstance(dense_activation_function, list):
        dense_activation_function = dense_activation_function[0]
    if not isinstance(dense_activation_function, str):
        raise TypeError(f'Expected type str for dense_activation_function, got: {type(dense_activation_function)}.')

    # LSTM history
    layer_input_history = Input(
        batch_input_shape=(batch_size, nr_timesteps_history, nr_features_history),
        name='history'
    )

    layer_lstm_history = None
    for i, (units, dropout) in enumerate(zip(list_units_per_layer, list_dropout_per_layer)):
        layer_lstm_history = LSTM(
            units,
            batch_input_shape=(1 if i == 0 else None, nr_timesteps_history, nr_features_history),
            dropout=dropout,
            stateful=stateful,
            use_bias=True,
            return_sequences=True if i < len(list_units_per_layer) - 1 else False,  # Last layer must not return sequences
            kernel_regularizer=kernel_regulizer,
            bias_regularizer=bias_regulizer
        )(layer_input_history if i == 0 else layer_lstm_history)

    # LSTM controlled
    layer_input_controlled = Input(
        batch_input_shape=(batch_size, nr_timesteps_prediction, nr_features_controlled),
        name='controlled')

    layer_lstm_controlled = LSTM(
        list_units_per_layer[-1],
        batch_input_shape=(batch_size, nr_timesteps_prediction, nr_features_controlled),
        dropout=list_dropout_per_layer[-1],
        stateful=stateful,
        use_bias=True,
        return_sequences=False,
        kernel_regularizer=kernel_regulizer,
        bias_regularizer=bias_regulizer
    )(layer_input_controlled)

    layer_combined = Concatenate(
        axis=1
    )([layer_lstm_history, layer_lstm_controlled])

    # Output layer
    layer_output = Dense(
        nr_timesteps_prediction * nr_features_prediction,
        activation=dense_activation_function
    )(layer_combined)

    # Reshape for correct nr of output features if multiple are to be predicted
    if nr_features_prediction > 1:
        layer_output = Reshape(
            target_shape=(nr_timesteps_prediction, nr_features_prediction)
        )(layer_output)

    # Assemble model
    model = Model(
        [layer_input_history, layer_input_controlled],
        layer_output,
        name=model_name
    )

    return model


def _create_model_gru(**kwargs):
    model_name = kwargs.get("model_name")
    list_units_per_layer = kwargs.get("list_units_per_layer")
    list_dropout_per_layer = kwargs.get("list_dropout_per_layer")
    stateful = kwargs.get("stateful")
    kernel_regulizer = kwargs.get("kernel_regulizer")
    bias_regulizer = kwargs.get("bias_regulizer")
    dense_activation_function = kwargs.get("dense_activation_function")
    batch_size = kwargs.get("batch_size")
    nr_timesteps_history = kwargs.get("nr_timesteps_history")
    nr_timesteps_prediction = kwargs.get("nr_timesteps_prediction")
    nr_features_history = kwargs.get("nr_features_history")
    nr_features_controlled = kwargs.get("nr_features_controlled")
    nr_features_prediction = kwargs.get("nr_features_prediction")

    if not isinstance(list_units_per_layer, list) or any(_ for _ in list_units_per_layer if not isinstance(_, int)):
        raise TypeError(f'Expected type list[int] for list_units_per_layer, got: {type(list_units_per_layer)}.')
    if len(list_units_per_layer) < 1:
        raise ValueError(f'Invalid value {list_units_per_layer} for list_units_per_layer given.')

    if not isinstance(list_dropout_per_layer, list) or any(_ for _ in list_dropout_per_layer if not isinstance(_, int)):
        raise TypeError(f'Expected type list[int] for list_dropout_per_layer, got: {type(list_dropout_per_layer)}.')
    if len(list_dropout_per_layer) != len(list_units_per_layer):
        raise ValueError(f'Invalid value {list_dropout_per_layer} for list_dropout_per_layer given.')

    if not isinstance(stateful, bool):
        raise TypeError(f'Expected type bool for stateful, got: {type(stateful)}.')

    if isinstance(kernel_regulizer, list):
        kernel_regulizer = kernel_regulizer[0]
    if not isinstance(kernel_regulizer, str):
        raise TypeError(f'Expected type str for kernel_regulizer, got: {type(kernel_regulizer)}.')

    if isinstance(bias_regulizer, list):
        bias_regulizer = bias_regulizer[0]
    if not isinstance(bias_regulizer, str):
        raise TypeError(f'Expected type str for bias_regulizer, got: {type(bias_regulizer)}.')

    if isinstance(dense_activation_function, list):
        dense_activation_function = dense_activation_function[0]
    if not isinstance(dense_activation_function, str):
        raise TypeError(f'Expected type str for dense_activation_function, got: {type(dense_activation_function)}.')

    # Input history
    layer_input_history = Input(
        batch_input_shape=(batch_size, nr_timesteps_history, nr_features_history),
        name='history'
    )

    layer_gru_history = None
    for i, (units, dropout) in enumerate(zip(list_units_per_layer, list_dropout_per_layer)):
        layer_gru_history = GRU(
            units,
            batch_input_shape=(1 if i == 0 else None, nr_timesteps_history, nr_features_history),
            dropout=dropout,
            stateful=stateful,
            use_bias=True,
            return_sequences=True if i < len(list_units_per_layer) - 1 else False,  # Last layer must not return sequences
            kernel_regularizer=kernel_regulizer,
            bias_regularizer=bias_regulizer,
            reset_after=True
        )(layer_input_history if i == 0 else layer_gru_history)

    # Controlled future branch
    layer_input_controlled = Input(
        batch_input_shape=(batch_size, nr_timesteps_prediction, nr_features_controlled),
        name='controlled')

    layer_gru_controlled = GRU(
        list_units_per_layer[-1],
        batch_input_shape=(batch_size, nr_timesteps_prediction, nr_features_controlled),
        dropout=list_dropout_per_layer[-1],
        stateful=stateful,
        use_bias=True,
        return_sequences=False,
        kernel_regularizer=kernel_regulizer,
        bias_regularizer=bias_regulizer,
        reset_after=True
    )(layer_input_controlled)

    layer_combined = Concatenate(
        axis=1
    )([layer_gru_history, layer_gru_controlled])

    # Output layer
    layer_output = Dense(
        nr_timesteps_prediction * nr_features_prediction,
        activation=dense_activation_function
    )(layer_combined)

    # Reshape for correct nr of output features if multiple are to be predicted
    if nr_features_prediction > 1:
        layer_output = Reshape(
            target_shape=(nr_timesteps_prediction, nr_features_prediction)
        )(layer_output)

    # Assemble model
    model = Model(
        [layer_input_history, layer_input_controlled],
        layer_output,
        name=model_name
    )

    return model


def _create_model_hybrid_lstm(**kwargs):
    nr_dps_per_minute = kwargs.get("nr_dps_per_minute")
    substance = kwargs.get("substance")
    density = kwargs.get("density")

    features_history = kwargs.get("features_history")
    features_prediction = kwargs.get("features_prediction")
    features_controlled = kwargs.get("features_controlled")
    features_hybrid = kwargs.get("features_hybrid")
    nr_features_history = kwargs.get("nr_features_history")
    nr_features_controlled = kwargs.get("nr_features_controlled")
    nr_features_hybrid = kwargs.get("nr_features_hybrid")

    model_name = kwargs.get("model_name")
    list_units_per_layer = kwargs.get("list_units_per_layer")
    list_dropout_per_layer = kwargs.get("list_dropout_per_layer")
    stateful = kwargs.get("stateful")
    kernel_regulizer = kwargs.get("kernel_regulizer")
    bias_regulizer = kwargs.get("bias_regulizer")
    dense_activation_function = kwargs.get("dense_activation_function")
    batch_size = kwargs.get("batch_size")
    nr_timesteps_history = kwargs.get("nr_timesteps_history")
    nr_timesteps_prediction = kwargs.get("nr_timesteps_prediction")

    h = kwargs.get("kwargs_hybrid", {}).get("h")
    l_max = kwargs.get("kwargs_hybrid", {}).get("l_max")
    l_critical = kwargs.get("kwargs_hybrid", {}).get("l_critical")
    k = kwargs.get("kwargs_hybrid", {}).get("k")

    if not isinstance(nr_dps_per_minute, int):
        raise TypeError(f'Expected type int for nr_dps_per_minute, got: {type(nr_dps_per_minute)}.')
    if not isinstance(substance, str):
        raise TypeError(f'Expected type float for substance, got: {type(substance)}.')
    if not isinstance(density, float):
        raise TypeError(f'Expected type float for substance, got: {type(density)}.')

    if not isinstance(features_history, list) or any(_ for _ in features_history if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_history, got: {type(features_history)}.')
    if not isinstance(features_prediction, list) or any(_ for _ in features_prediction if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_prediction, got: {type(features_prediction)}.')
    if not isinstance(features_controlled, list) or any(_ for _ in features_controlled if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_controlled, got: {type(features_controlled)}.')
    if not isinstance(features_hybrid, list) or any(_ for _ in features_hybrid if not isinstance(_, str)):
        raise TypeError(f'Expected type list[str] for features_hybrid, got: {type(features_hybrid)}.')

    if not isinstance(nr_features_hybrid, int):
        raise TypeError(f'Expected type int for nr_features_hybrid, got: {type(nr_features_hybrid)}.')

    if not isinstance(list_units_per_layer, list) or any(_ for _ in list_units_per_layer if not isinstance(_, int)):
        raise TypeError(f'Expected type list[int] for list_units_per_layer, got: {type(list_units_per_layer)}.')
    if len(list_units_per_layer) < 1:
        raise ValueError(f'Invalid value {list_units_per_layer} for list_units_per_layer given.')

    if not isinstance(list_dropout_per_layer, list) or any(_ for _ in list_dropout_per_layer if not isinstance(_, int)):
        raise TypeError(f'Expected type list[int] for list_dropout_per_layer, got: {type(list_dropout_per_layer)}.')
    if len(list_dropout_per_layer) != len(list_units_per_layer):
        raise ValueError(f'Invalid value {list_dropout_per_layer} for list_dropout_per_layer given.')

    if not isinstance(stateful, bool):
        raise TypeError(f'Expected type bool for stateful, got: {type(stateful)}.')

    if isinstance(kernel_regulizer, list):
        kernel_regulizer = kernel_regulizer[0]
    if not isinstance(kernel_regulizer, str):
        raise TypeError(f'Expected type str for kernel_regulizer, got: {type(kernel_regulizer)}.')

    if isinstance(bias_regulizer, list):
        bias_regulizer = bias_regulizer[0]
    if not isinstance(bias_regulizer, str):
        raise TypeError(f'Expected type str for bias_regulizer, got: {type(bias_regulizer)}.')

    if isinstance(dense_activation_function, list):
        dense_activation_function = dense_activation_function[0]
    if not isinstance(dense_activation_function, str):
        raise TypeError(f'Expected type str for dense_activation_function, got: {type(dense_activation_function)}.')

    if not isinstance(h, float | int):
        raise TypeError(f'Expected type float | int for h, got: {type(h)}.')
    if not isinstance(l_max, float | int):
        raise TypeError(f'Expected type float | int for l_max, got: {type(l_max)}.')
    if not isinstance(l_critical, float | int):
        raise TypeError(f'Expected type float | int for l_critical, got: {type(l_critical)}.')
    if not isinstance(k, float | int):
        raise TypeError(f'Expected type float | int for k, got: {type(k)}.')

    # LSTM history
    layer_input_history = Input(
        batch_input_shape=(batch_size, nr_timesteps_history, nr_features_history),
        name='history'
    )

    layer_lstm_history = None
    for i, (units, dropout) in enumerate(zip(list_units_per_layer, list_dropout_per_layer)):
        layer_lstm_history = LSTM(
            units,
            batch_input_shape=(1 if i == 0 else None, nr_timesteps_history, nr_features_history),
            dropout=dropout,
            stateful=stateful,
            use_bias=True,
            return_sequences=True if i < len(list_units_per_layer) - 1 else False,  # Last layer must not return sequences
            kernel_regularizer=kernel_regulizer,
            bias_regularizer=bias_regulizer
        )(layer_input_history if i == 0 else layer_lstm_history)

    # LSTM controlled
    layer_input_controlled = Input(
        batch_input_shape=(batch_size, nr_timesteps_prediction, nr_features_controlled),
        name='controlled')
    layer_lstm_controlled = LSTM(
        list_units_per_layer[-1],
        batch_input_shape=(batch_size, nr_timesteps_prediction, nr_features_controlled),
        dropout=list_dropout_per_layer[-1],
        stateful=stateful,
        use_bias=True,
        return_sequences=False,
        kernel_regularizer=kernel_regulizer,
        bias_regularizer=bias_regulizer
    )(layer_input_controlled)

    layer_combined = Concatenate(
        axis=1
    )([layer_lstm_history, layer_lstm_controlled])

    # Output layer
    layer_output_lstm = Dense(
        nr_features_hybrid,
        activation=dense_activation_function,
        name='output'
    )(layer_combined)

    print("ANN model created")

    kwargs_pbe = {
        "features_history": features_history,
        "features_controlled": features_controlled,
        "features_prediction": features_prediction,
        "features_hybrid": features_hybrid,
        "nr_features_history": nr_features_history,
        "nr_features_controlled": nr_features_controlled,
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

    # Assemble model
    model = ModelPBE(
        inputs=[layer_input_history, layer_input_controlled],
        outputs=layer_output_lstm,  # The loss func is called seperately for each of the list items
        name=model_name
    )

    model.init_pbe(kwargs_pbe)

    print("model assembled")
    return model


if __name__ == '__main__':
    create_model()
