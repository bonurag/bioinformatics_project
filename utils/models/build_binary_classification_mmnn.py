from typing import Optional
import silence_tensorflow.auto
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.python.layers.base import Layer

from extra_keras_metrics import get_complete_binary_metrics

from utils.bio_constants import MMNN_SIMPLE, MMNN_BOOST
from utils.models.build_binary_classification_cnn import build_binary_classification_cnn
from utils.models.build_binary_classification_ffnn import build_binary_classification_ffnn


def build_binary_classification_mmnn(
        hp_param_mmnn: dict(),
        hp_param_ffnn: Optional[dict] = None,
        hp_param_cnn: Optional[dict] = None,
        input_shape: Optional[int] = None,
        window_size: Optional[int] = None,
        input_epigenomic_data: Optional[Layer] = None,
        input_sequence_data: Optional[Layer] = None,
        last_hidden_ffnn: Optional[Layer] = None,
        last_hidden_cnn: Optional[Layer] = None,
) -> Model:
    """Returns Multi-Modal Neural Network model for binary classification.

    Implementative details
    -----------------------
    If the input shape / window size is not provided and the input layers and
    the feature selection layers are provided, then the network will start
    to train from those layers (which are expected to be pre-trained).
    Conversely, it will create the submodules for the epigenomic and sequence
    data ex-novo.

    Parameters
    -----------------------
    input_shape: Optional[int] = None,
        Number of features in the input layer.
        Either the input shape or the input and output layers of the FFNN
        must be provided.
    window_size: int,
        Size of the input genomic window.
        Either the window size or the input and output layers of the CNN
        must be provided.
    input_epigenomic_data: Optional[Layer] = None,
        Input for the epigenomic data from a FFNN model.
        Either the input shape or the input and output layers of the FFNN
        must be provided.
    input_sequence_data: Optional[Layer] = None,
        Input for the sequence data from a CNN model.
        Either the window size or the input and output layers of the CNN
        must be provided.
    last_hidden_ffnn: Optional[Layer] = None,
        Feature selection layer from a FFNN model.
        Either the input shape or the input and output layers of the FFNN
        must be provided.
    last_hidden_cnn: Optional[Layer] = None,
        Feature selection layer from a CNN model.
        Either the window size or the input and output layers of the CNN
        must be provided.

    Raises
    -----------------------
    ValueError,
        If the input shape is not provided and the input layer and feature selection
        layer of the FFNN are not provided.
    ValueError,
        If the window size is not provided and the input layer and feature selection
        layer of the CNN are not provided.

    Returns
    -----------------------
    Triple with model, input layer and output layer.
    """

    learning_rate = hp_param_mmnn.get("learning_rate")
    n_neurons_concat = hp_param_mmnn.get("n_neurons_concat")

    if input_shape is None and (last_hidden_ffnn is None or input_epigenomic_data is None):
        raise ValueError(
            "Either the input shape or the features selection layer and the input epigenomic "
            "layer must be provided."
        )
    if window_size is None and (last_hidden_cnn is None or input_sequence_data is None):
        raise ValueError(
            "Either the input shape or the features selection layer and the input sequence "
            "layer must be provided."
        )

    if input_shape is not None and hp_param_ffnn is not None:
        _, input_epigenomic_data, last_hidden_ffnn = build_binary_classification_ffnn(input_shape, hp_param_ffnn)

    if window_size is not None and hp_param_cnn is not None:
        _, input_sequence_data, last_hidden_cnn = build_binary_classification_cnn(window_size, hp_param_cnn)

    concatenation_layer = Concatenate()([
        last_hidden_ffnn,
        last_hidden_cnn
    ])

    last_hidden_mmnn = Dense(n_neurons_concat, activation="relu", name="last_hidden_mmnn")(concatenation_layer)
    output_mmnn = Dense(1, activation="sigmoid")(last_hidden_mmnn)

    mmnn = Model(
        inputs=[input_epigenomic_data, input_sequence_data],
        outputs=output_mmnn,
        name=MMNN_BOOST if input_shape is None else MMNN_SIMPLE
    )

    mmnn.compile(
        optimizer=optimizers.Nadam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=get_complete_binary_metrics()
    )

    return mmnn