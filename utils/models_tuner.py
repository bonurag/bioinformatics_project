from ucsc_genomes_downloader import Genome

from typing import Tuple
from utils.hypermodels import ffnn_hypermodel, cnn_hypermodel, mmnn_hypermodel
from utils.data_processing import get_ffnn_sequence, get_cnn_sequence, get_mmnn_sequence
from utils.bio_constants import *

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
#from loguru import logger
from typing import Optional

from keras_tuner import Hyperband
import keras_tuner as kt

import numpy as np
import pandas as pd

from datetime import datetime
import datetime


def hyperparameter_tuning(
        train_X: np.ndarray,
        test_X: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
        train_bed: pd.DataFrame,
        test_bed: pd.DataFrame,
        genome: Genome,
        window_size: int,
        holdout_number: int,
        task_name: str,
        model_name: str,
        input_layers: Optional[list] = None,
        hidden_layers: Optional[list] = None
)-> Tuple[Model, Layer, Layer]:
    """Returns tuple with list of kept features and list of discared features.

    Parameters
    --------------------------
    train_X: np.ndarray,
        The vector from where to extract the epigenomic. Used for training phase.
    test_X: np.ndarray,
        The vector from where to extract the epigenomic. Used for test phase.
    train_y: np.ndarray,
        The values the model should predict during the training phase.
    test_y: np.ndarray,
        The values the model should predict during the test phase.
    holdout_number: int,
        The current holdout number.
    task_name: str,
        The name of the task.
    model_name: str,
        The name of the model.

    Returns
    -------
    Return FFNN model resulting from hyperparameter optimization, and dictionary
    that contains best hyperparamters results.
    """

    global hyperparam_results, max_epochs, project_name, directory

    task_name = "AEvsIE" if task_name == "active_enhancers_vs_inactive_enhancers" else "APvsIP"

    if model_name == MODEL_TYPE_FFNN:
        hypermodel = ffnn_hypermodel.FFNNHyperModel(input_shape=train_X.shape[1])
        max_epochs = HP_MAX_EPOCHS_FFNN
        project_name = TUNER_PROJECT_NAME_FFNN
        directory = TUNER_DIR_FFNN
    elif model_name == MODEL_TYPE_CNN:
        hypermodel = cnn_hypermodel.CNNHyperModel(window_size)
        max_epochs = HP_MAX_EPOCHS_CNN
        project_name = TUNER_PROJECT_NAME_CNN
        directory = TUNER_DIR_CNN
    elif model_name == MODEL_TYPE_MMNN:
        if input_layers and hidden_layers:
            print(f"model_name: {model_name} input_layers: {input_layers} hidden_layers: {hidden_layers}")
            print(input_layers.get("input_epigenomic_data"))
            hypermodel = mmnn_hypermodel.MMNNHyperModel(input_layers.get("input_epigenomic_data"),
                                                        input_layers.get("input_sequence_data"),
                                                        hidden_layers.get("last_hidden_ffnn"),
                                                        hidden_layers.get("last_hidden_cnn")
                                                        )
        max_epochs = HP_MAX_EPOCHS_MMNN
        project_name = TUNER_PROJECT_NAME_MMNN
        directory = TUNER_DIR_MMNN

    tuners = define_tuners(hypermodel,
                            max_epochs=max_epochs,
                            directory=directory,
                            project_name=project_name)

    for tuner in tuners:
        hyperparam_results = tuner_evaluation(tuner,
                                                train_X,
                                                test_X,
                                                train_y,
                                                test_y,
                                                train_bed,
                                                test_bed,
                                                genome,
                                                task_name,
                                                holdout_number,
                                                model_name
    )

    return hyperparam_results


def tuner_evaluation(tuner, train_X, test_X, train_y, test_y, train_bed, test_bed, genome, task_name, holdout_number, model_name):
    # Overview of the task
    # tuner.search_space_summary()

    # Performs the hyperparameter tuning
    global train_search_seq, valid_search_seq
    model = None

    #logger.info(f"Start hyperparameter tuning for {model_name}")

    if model_name == MODEL_TYPE_FFNN:
        train_search_seq = get_ffnn_sequence(train_X, train_y)
        valid_search_seq = get_ffnn_sequence(test_X, test_y)
    elif model_name == MODEL_TYPE_CNN:
        train_search_seq = get_cnn_sequence(genome, train_bed, train_y)
        valid_search_seq = get_cnn_sequence(genome, test_bed, test_y)
    elif model_name == MODEL_TYPE_MMNN:
        train_search_seq = get_mmnn_sequence(genome, train_bed, train_X, train_y)
        valid_search_seq = get_mmnn_sequence(genome, test_bed, test_X, test_y)

    tuner.search(train_search_seq,
                    validation_data=valid_search_seq,
                    epochs=200,
                    batch_size=256,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=3, verbose=1)]
    )

    # Show a summary of the search
    # tuner.results_summary()

    # Retrieve the best hyperparameters.
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    if model_name == MODEL_TYPE_FFNN:
        print(f"Get Layer From {model_name} Models!")
        input_epigenomic_data = model[1]
        last_hidden_ffnn = model[2]

    if model_name == MODEL_TYPE_CNN:
        print(f"Get Layer From {model_name} Models!")
        input_sequence_data = model[1]
        last_hidden_cnn = model[2]

    results = {
        "task_name": task_name,
        "holdout_number": holdout_number,
        "learning_rate": best_hps.get("learning_rate"),
        "create_date": datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    }

    if model_name == MODEL_TYPE_FFNN:
        results["num_layers"] = best_hps.get("num_layers")
        results["n_neurons0"] = best_hps.get("n_neurons0")
        results["n_neurons1"] = best_hps.get("n_neurons1") if best_hps.get("num_layers") >= 3 else None
        results["input_epigenomic_data"] = input_epigenomic_data
        results["last_hidden_ffnn"] = last_hidden_ffnn

    if model_name == MODEL_TYPE_CNN:
        results["num_layers"] = best_hps.get("num_conv_layers")
        results["n_neurons0"] = best_hps.get("n_neurons0")
        results["kernel_size0"] = best_hps.get("kernel_size0")
        results["drop_rate"] = best_hps.get("drop_rate")
        results["n_neurons1"] = best_hps.get("n_neurons1") if best_hps.get("num_conv_layers") >= 3 else None
        results["kernel_size1"] = best_hps.get("kernel_size1") if best_hps.get("num_conv_layers") >= 3 else None
        results["n_neurons_last_out"] = best_hps.get("n_neurons_last_out")
        results["drop_rate_out"] = best_hps.get("drop_rate_out")
        results["input_sequence_data"] = input_sequence_data
        results["last_hidden_cnn"] = last_hidden_cnn

    if model_name == MODEL_TYPE_MMNN:
        results["n_neurons_concat"] = best_hps.get("n_neurons_concat")

    return {model_name: model, f"{model_name}_parameters": results}


def define_tuners(hypermodel, max_epochs, directory, project_name):
    hyperband_tuner = Hyperband(
        hypermodel,
        max_epochs=max_epochs,
        objective=kt.Objective("val_AUPRC", direction="max"),
        factor=3,
        directory=directory,
        project_name=project_name
    )
    return [hyperband_tuner]
