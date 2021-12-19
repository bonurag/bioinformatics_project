from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras import optimizers
from extra_keras_metrics import get_complete_binary_metrics
from kerastuner import HyperModel

from utils.bio_constants import CNN_NAME


class CNNHyperModel(HyperModel):
    def __init__(self, window_size):
        self.window_size = window_size

    def build(self, hp):
        num_conv_layers = hp.Int(name="num_conv_layers", min_value=2, max_value=8, step=1)
        n_neurons0 = hp.Int(name="n_neurons0", min_value=32, max_value=128, step=32)
        kernel_size0 = hp.Int(name="kernel_size0", min_value=5, max_value=8)
        drop_rate = hp.Float(name="drop_rate", min_value=0.0, max_value=0.5)
        drop_rate_out = hp.Float(name="drop_rate_out", min_value=0.0, max_value=0.5)
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-4])
        n_neurons_last_out = hp.Int(name="n_neurons_last_out", min_value=16, max_value=128, step=16)

        input_sequence_data = Input((self.window_size, 4), name="input_sequence_data")
        hidden = input_sequence_data

        for num_conv_layer in range(num_conv_layers):
            if num_conv_layer >= 2:
                with hp.conditional_scope("num_conv_layers", [3, 4, 5, 6, 7, 8]):
                    n_neurons1 = hp.Int(name="n_neurons1", min_value=16, max_value=128, step=16)
                    kernel_size1 = hp.Int(name="kernel_size1", min_value=2, max_value=10)

                    hidden = Conv1D(n_neurons1, kernel_size=kernel_size1)(hidden)
                    hidden = BatchNormalization()(hidden)
                    hidden = ReLU()(hidden)
                    hidden = Dropout(drop_rate)(hidden)
            else:
                hidden = Conv1D(n_neurons0, kernel_size=kernel_size0)(hidden)
                hidden = BatchNormalization()(hidden)
                hidden = ReLU()(hidden)
                hidden = Dropout(drop_rate)(hidden)

            if num_conv_layer % 2 != 0:
                hidden = MaxPool1D(pool_size=2, padding='same')(hidden)

        hidden = GlobalAveragePooling1D()(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(drop_rate_out)(hidden)
        last_hidden_cnn = Dense(n_neurons_last_out, activation="relu", name="last_hidden_cnn")(hidden)
        output_cnn = Dense(1, activation="sigmoid", name="output_cnn")(last_hidden_cnn)

        model = Model(inputs=input_sequence_data, outputs=output_cnn, name=CNN_NAME)

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Nadam(learning_rate=learning_rate),
            metrics=get_complete_binary_metrics()
        )

        return model, input_sequence_data, last_hidden_cnn
