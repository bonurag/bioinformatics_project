from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
from extra_keras_metrics import get_complete_binary_metrics
from keras_tuner import HyperModel

from utils.bio_constants import FFNN_NAME_HP


class FFNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        num_layers = hp.Int(name="num_layers", min_value=2, max_value=6)
        n_neurons0 = hp.Int(name="n_neurons0", min_value=32, max_value=256, step=32)
        learning_rate = hp.Choice(name="learning_rate", values=[1e-2, 1e-4])

        input_epigenomic_data = Input(shape=(self.input_shape,), name="input_epigenomic_data")
        last_hidden_ffnn = hidden = input_epigenomic_data

        for layer in range(num_layers):
            if layer == (num_layers - 1):
                name = "last_hidden_ffnn"
            else:
                name = None
            if layer >= 2:
                with hp.conditional_scope("num_layers", [3, 4, 5, 6, 6]):
                    n_neurons1 = hp.Int(
                        name="n_neurons1",
                        min_value=16,
                        max_value=256,
                        step=16
                    )

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

        model = Model(inputs=input_epigenomic_data, outputs=output_ffnn, name=FFNN_NAME_HP)

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Nadam(learning_rate=learning_rate),
            metrics=get_complete_binary_metrics()
        )

        return model
