from typing import Tuple
import silence_tensorflow.auto
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras import optimizers
from extra_keras_metrics import get_complete_binary_metrics

from utils.bio_constants import FFNN_NAME


def build_binary_classification_ffnn(
    input_shape: int,
    hp_param: dict()
) -> Tuple[Model, Layer, Layer]:
    """Build a custom Feed-Forward Neural Network.

    Parameters
    ----------
    input_shape: int,
        Number of features in the input layer.
    hp_param : dict
        Dictionary with best hyperparameters used for buil net.

    Returns
    -------
    The compiled FFNN.
    """

    num_layers = hp_param.get('num_layers')
    n_neurons0 = hp_param.get('n_neurons0')
    n_neurons1 = hp_param.get('n_neurons1')
    learning_rate = hp_param.get('learning_rate')

    input_epigenomic_data = Input(shape=(input_shape,), name="input_epigenomic_data")
    last_hidden_ffnn = hidden = input_epigenomic_data

    for layer in range(num_layers):
        if layer == (num_layers - 1):
            name = "last_hidden_ffnn"
        else:
            name = None
        if layer >= 2:
            hidden = Dense(
                n_neurons1,
                activation="relu",
                kernel_regularizer=None,
                name=name
            )(hidden)
            last_hidden_ffnn = hidden
        else:
            hidden = Dense(
                n_neurons0,
                activation="relu",
                kernel_regularizer=None,
                name=name
            )(hidden)
            last_hidden_ffnn = hidden

    output_ffnn = Dense(1, activation="sigmoid")(last_hidden_ffnn)

    ffnn = Model(
        inputs=input_epigenomic_data,
        outputs=output_ffnn,
        name=FFNN_NAME
    )

    ffnn.compile(
        optimizer=optimizers.Nadam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=get_complete_binary_metrics()
    )

    return ffnn, input_epigenomic_data, last_hidden_ffnn