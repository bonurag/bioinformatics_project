from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from keras_mixed_sequence import MixedSequence, VectorSequence
from keras_bed_sequence import BedSequence
from cache_decorator import Cache
from multiprocessing import cpu_count
from ucsc_genomes_downloader import Genome
from utils.bio_constants import GENOME_CACHE_DIR

import pandas as pd
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from typing import Dict, List, Tuple, Optional, Union


from utils.data_processing import *

from pathlib import Path
from datetime import datetime
import time


def get_cnn_sequence(
        genome: Genome,
        bed: pd.DataFrame,
        y: np.ndarray,
        batch_size: int = 1024
) -> MixedSequence:
    """Returns sequence to train a CNN model on genomic sequences.

    Implementative details
    -------------------------
    This sequence can be used for either binary classification or
    for regresssion, just change the y accordingly.

    Parameters
    -------------------------
    genome: Genome,
        The genome from where to extract the genomic sequence.
    bed: pd.DataFrame,
        The BED file coordinates describing where to extract the sequences.
    y: np.ndarray,
        The values the model should predict.
    batch_size: int = 1024,
        The size of the batches to generate

    Returns
    --------------------------
    MixedSequence object to train a CNN.
    """
    return MixedSequence(
        x={
            "input_sequence_data": BedSequence(
                genome,
                bed,
                batch_size=batch_size,
            )
        },
        y=VectorSequence(
            y,
            batch_size=batch_size
        )
    )


def get_ffnn_sequence(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 1024
) -> MixedSequence:
    """Returns sequence to train a FFNN model on epigenomic data.

    Implementative details
    -------------------------
    This sequence can be used for either binary classification or
    for regresssion, just change the y accordingly.

    Parameters
    -------------------------
    X: np.ndarray,
        The vector from where to extract the epigenomic data.
    y: np.ndarray,
        The values the model should predict.
    batch_size: int = 1024,
        The size of the batches to generate

    Returns
    --------------------------
    MixedSequence object to train a FFNN.
    """
    return MixedSequence(
        x={
            "input_epigenomic_data": VectorSequence(
                X,
                batch_size
            )
        },
        y=VectorSequence(
            y,
            batch_size=batch_size
        )
    )


def get_mmnn_sequence(
        genome: Genome,
        bed: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 1024
) -> MixedSequence:
    """Returns sequence to train a MMNN model on both genomic sequences and epigenomic data.

    Implementative details
    -------------------------
    This sequence can be used for either binary classification or
    for regresssion, just change the y accordingly.

    Parameters
    -------------------------
    genome: Genome,
        The genome from where to extract the genomic sequence.
    bed: pd.DataFrame,
        The BED file coordinates describing where to extract the sequences.
    X: np.ndarray,
        The vector from where to extract the epigenomic data.
    y: np.ndarray,
        The values the model should predict.
    batch_size: int = 1024,
        The size of the batches to generate

    Returns
    --------------------------
    MixedSequence object to train a MMNN.
    """
    return MixedSequence(
        x={
            "input_sequence_data": BedSequence(
                genome,
                bed,
                batch_size=batch_size,
            ),
            "input_epigenomic_data": VectorSequence(
                X,
                batch_size
            )
        },
        y=VectorSequence(
            y,
            batch_size=batch_size
        )
    )


@Cache(
    cache_path=[
        "boruta/ffnn/kept_features_{_hash}.json",
        "boruta/ffnn/discarded_features_{_hash}.json",
        "boruta/cnn/kept_features_{_hash}.json",
        "boruta/cnn/discarded_features_{_hash}.json",
        "boruta/mmnn/kept_features_{_hash}.json",
        "boruta/mmnn/discarded_features_{_hash}.json"
    ],
    args_to_ignore=["X_train", "y_train"]
)
def execute_boruta_feature_selection(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        holdout_number: int,
        task_name: str,
        max_iter: int = 100
):
    """Returns tuple with list of kept features and list of discared features.
    
    Parameters
    --------------------------
    X_train: pd.DataFrame,
        The data reserved for the input of the training of the Boruta model.
    y_train: np.ndarray,
        The data reserved for the output of the training of the Boruta model.
    holdout_number: int,
        The current holdout number.
    task_name: str,
        The name of the task.
    max_iter: int = 100,
        Number of iterations to run Boruta for.

    Returns
    -------
    kept_features: list(),
        List of indices referring to the features to be maintained.
    discarded_features: list(),
        List of indices referring to the features to be eliminated.
    """

    model = RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced_subsample', max_depth=5)

    # Create the Boruta model
    boruta_selector = BorutaPy(
        model,  # Defining the model that Boruta should use.
        n_estimators='auto',  # We leave the number of estimators to be decided by Boruta.
        verbose=False,
        alpha=0.05,  # p_value
        # In practice one would run at least 100-200 times,
        # until all tentative features are exausted.
        max_iter=max_iter,
        random_state=42,
    )
    # Fit the Boruta model
    boruta_selector.fit(X_train.values, y_train)

    # Get the kept features and discarded features
    kept_features = list(X_train.columns[boruta_selector.support_])
    discarded_features = list(X_train.columns[~boruta_selector.support_])

    # Filter out the unused featured.
    return kept_features, discarded_features


def to_bed(data: pd.DataFrame) -> pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]


def get_genome() -> Genome:
    return Genome("hg38", cache_directory=GENOME_CACHE_DIR)
