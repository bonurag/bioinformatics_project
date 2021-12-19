from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from extra_keras_metrics import get_complete_binary_metrics
from kerastuner import HyperModel

from utils.bio_constants import MMNN_NAME


class MMNNHyperModel(HyperModel):
    def __init__(self, input_epigenomic_data, input_sequence_data, last_hidden_ffnn, last_hidden_cnn):
        self.input_epigenomic_data = input_epigenomic_data
        self.input_sequence_data = input_sequence_data
        self.last_hidden_ffnn = last_hidden_ffnn
        self.last_hidden_cnn = last_hidden_cnn

    def build(self, hp):
        n_neurons_concat = hp.Int(name="n_neurons_concat", min_value=32, max_value=256, step=32)
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-4])

        concatenation_layer = Concatenate()([self.last_hidden_ffnn, self.last_hidden_cnn])

        last_hidden_mmnn = Dense(n_neurons_concat, activation="relu", name="First_Hidden_Layer")(concatenation_layer)
        output_mmnn = Dense(1, activation="sigmoid", name="Output_Layer")(last_hidden_mmnn)

        model = Model(inputs=[self.input_epigenomic_data, self.input_sequence_data], outputs=output_mmnn, name=MMNN_NAME)
        model.compile(
            optimizer=optimizers.Nadam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=get_complete_binary_metrics()
        )

        return model
