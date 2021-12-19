from cache_decorator import Cache

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from typing import Dict, List, Tuple, Optional, Union

from keras_mixed_sequence import MixedSequence

from utils.data_processing import *

from pathlib import Path
from datetime import datetime
import time

@Cache(
    cache_path=[
        "model_histories/{cell_line}/{task}/{model_name}/{use_feature_selection}/history_{_hash}.csv.xz",
        "model_performance/{cell_line}/{task}/{model_name}/{use_feature_selection}/performance_{_hash}.csv.xz",
        "model_histories/{cell_line}/{task}/{model_name}/{use_feature_selection}/history_{_hash}.csv.xz",
        "model_performance/{cell_line}/{task}/{model_name}/{use_feature_selection}/performance_{_hash}.csv.xz",
        "model_histories/{cell_line}/{task}/{model_name}/{use_feature_selection}/history_{_hash}.csv.xz",
        "model_performance/{cell_line}/{task}/{model_name}/{use_feature_selection}/performance_{_hash}.csv.xz"
    ],
    args_to_ignore=[
        "model", "training_sequence", "test_sequence"
    ]
)
def train_model(
    model: Model,
    model_name: str,
    task: str,
    cell_line: str,
    training_sequence: MixedSequence,
    test_sequence: MixedSequence,
    holdout_number: int,
    use_feature_selection: bool,
    start_time: time) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns training history and model evaluations.
    
    Parameters
    ---------------------
    model: Model,
        The model to train.
    model_name: str,
        The model name.
    task: str,
        The name of the task.
    cell_line: str,
        Name of the considered cell line.
    training_sequence: MixedSequence,
        The training sequence.
    test_sequence: MixedSequence,
        The test sequence.
    holdout_number: int,
        The number of the current holdout.
    use_feature_selection: bool,
        Whether the model is trained using features that have
        been selected with Boruta or not.

    Returns
    ----------------------
    Tuple with training history dataframe and model evaluations dataframe.
    """
    history = pd.DataFrame(model.fit(
        train_sequence,
        validation_data=test_sequence,
        epochs=1000,
        verbose=False,
        callbacks=[
            EarlyStopping(
                "loss",
                min_delta=0.001,
                patience=2,
                mode="min"
            )
            #TqdmCallback(verbose=1)
        ]
    ).history)
    
    train_evaluation = dict(zip(model.metrics_names, model.evaluate(train_sequence, verbose=False)))
    test_evaluation = dict(zip(model.metrics_names, model.evaluate(test_sequence, verbose=False)))
    train_evaluation["run_type"] = "train"
    test_evaluation["run_type"] = "test"
    
    for evaluation in (train_evaluation, test_evaluation):
        evaluation["model_name"] = model_name
        evaluation["task"] = task
        evaluation["holdout_number"] = holdout_number 
        evaluation["use_feature_selection"] = use_feature_selection
        evaluation["elapsed_time"] = round(time.time() - start_time, 2)

    evaluations = pd.DataFrame([
        train_evaluation,
        test_evaluation
    ])
    
    return history, evaluations
