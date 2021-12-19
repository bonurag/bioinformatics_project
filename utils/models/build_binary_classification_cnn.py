from typing import Tuple
import silence_tensorflow.auto
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras import optimizers
from tensorflow.python.layers.base import Layer
from extra_keras_metrics import get_complete_binary_metrics

from utils.bio_constants import CNN_NAME

def build_binary_classification_cnn(
        window_size: int,
        hp_param: dict()
) -> Tuple[Model, Layer, Layer]:
    """Returns Convolutional Neural Network model for binary classification.

    Parameters
    -----------------------
    window_size: int,
        Size of the input genomic window.

    Returns
    -----------------------
    Triple with model, input layer and output layer.
    """

    num_layers = hp_param.get("num_layers")
    n_neurons0 = hp_param.get("n_neurons0")
    kernel_size0 = hp_param.get("kernel_size0")

    n_neurons1 = hp_param.get("n_neurons1")
    kernel_size1 = hp_param.get("kernel_size1")

    drop_rate = hp_param.get("drop_rate")

    n_neurons_last_out = hp_param.get("n_neurons_last_out")
    drop_rate_out = hp_param.get("drop_rate_out")

    learning_rate = hp_param.get("learning_rate")

    input_sequence_data = Input(shape=(window_size, 4), name="input_sequence_data")
    hidden = Conv1D(n_neurons0, kernel_size=kernel_size0, activation="relu")(input_sequence_data)

    for _ in range(num_layers):
        hidden = Conv1D(
            n_neurons1,
            kernel_size=kernel_size1,
            activation="relu",
        )(hidden)
        hidden = Dropout(rate=drop_rate)(hidden)
        hidden = MaxPool1D(pool_size=2, padding='same')(hidden)

    hidden = GlobalAveragePooling1D()(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(rate=drop_rate_out)(hidden)
    last_hidden_cnn = Dense(n_neurons_last_out, activation="relu")(hidden)
    output_cnn = Dense(1, activation="sigmoid")(last_hidden_cnn)

    cnn = Model(
        inputs=input_sequence_data,
        outputs=output_cnn,
        name=CNN_NAME
    )

    cnn.compile(
        optimizer=optimizers.Nadam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=get_complete_binary_metrics()
    )
    return cnn, input_sequence_data, last_hidden_cnn