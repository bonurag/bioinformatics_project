import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from cache_decorator import Cache
from multiprocessing import cpu_count

from boruta import BorutaPy

from keras_mixed_sequence import MixedSequence, VectorSequence
from keras_bed_sequence import BedSequence

from ucsc_genomes_downloader import Genome

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as UTSNE

from barplots import barplots

from utils.data_processing import *
from utils.bio_constants import GENOME_CACHE_DIR, WINDOW_SIZE

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
        "boruta/kept_features_{_hash}.json",
        "boruta/discarded_features_{_hash}.json"
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


def one_hot_encode(genome:Genome, data:pd.DataFrame, nucleotides:str="actg")->np.ndarray:
    return np.array(BedSequence(
        genome,
        bed=to_bed(data),
        nucleotides=nucleotides,
        batch_size=1
    ))


def flat_one_hot_encode(genome:Genome, data:pd.DataFrame, window_size:int, nucleotides:str="actg")->np.ndarray:
    return one_hot_encode(genome, data, nucleotides).reshape(-1, WINDOW_SIZE*4).astype(int)


def to_dataframe(x:np.ndarray, window_size:int, nucleotides:str="actg")->pd.DataFrame:
    return pd.DataFrame(
        x,
        columns = [
            f"{i}{nucleotide}"
            for i in range(WINDOW_SIZE)
            for nucleotide in nucleotides
        ]
    )


def drop_constant_features(df:pd.DataFrame)->pd.DataFrame:
    """Return DataFrame without constant features."""
    return df.loc[:, (df != df.iloc[0]).any()]


def knn_imputation(df:pd.DataFrame, neighbours:int=5)->pd.DataFrame:
    """Return provided dataframe with NaN imputed using knn.

    Parameters
    --------------------
    df:pd.DataFrame,
        The dataframe to impute.
    neighbours:int=5,
        The number of neighbours to consider.

    Returns
    --------------------
    The dataframe with the NaN values imputed.
    """
    return pd.DataFrame(
        KNNImputer(n_neighbors=neighbours).fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )

def robust_zscoring(df:pd.DataFrame)->pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )


def get_top_most_different(dist, n:int):
    return np.argsort(-np.mean(dist, axis=1).flatten())[:n]


def get_features_filter(X:pd.DataFrame, y:pd.DataFrame, name:str)->BorutaPy:
    boruta_selector = BorutaPy(
        RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5),
        n_estimators='auto',
        verbose=2,
        alpha=0.05, # p_value
        max_iter=30, # In practice one would run at least 100-200 times
        random_state=42
    )
    boruta_selector.fit(X.values, y.values.ravel())
    return boruta_selector


def pca(x:np.ndarray, n_components:int=2)->np.ndarray:
    return PCA(n_components=n_components, random_state=42).fit_transform(x)


def ulyanov_tsne(
    x: np.ndarray,
    perplexity: int,
    dimensionality_threshold: int = 50,
    n_components: int = 2
):
    if x.shape[1] > dimensionality_threshold:
        x = pca(x, n_components=dimensionality_threshold)
    return UTSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_jobs=cpu_count(),
        random_state=42,
        verbose=True
    ).fit_transform(x)


def get_genome() -> Genome:
    return Genome("hg38", cache_directory=GENOME_CACHE_DIR)


def normalize_epigenomic_data(
    train_x: np.ndarray,
    test_x: np.ndarray = None
) -> Tuple[np.ndarray]:
    """Return imputed and normalized epigenomic data.

    We fit the imputation and normalization on the training data and
    apply it to both the training data and the test data.

    Parameters
    -------------------------
    train_x: np.ndarray,
        Training data to use to fit the imputer and scaled.
    test_x: np.ndarray = None,
        Test data to be normalized.

    Returns
    -------------------------
    Tuple with imputed and scaled train and test data.
    """
    # Create the imputer and scaler object
    imputer = KNNImputer()
    scaler = RobustScaler()
    # Fit the imputer object
    imputer.fit(train_x)
    # Impute the train and test data
    imputed_train_x = imputer.transform(train_x)
    if test_x is not None:
        imputed_test_x = imputer.transform(test_x)
    # Fit the scaler object
    scaler.fit(imputed_train_x)
    # Scale the train and test data
    scaled_train_x = scaler.transform(imputed_train_x)
    if test_x is not None:
        scaled_test_x = scaler.transform(imputed_test_x)
    if test_x is not None:
        # Return the normalized data
        return scaled_train_x, scaled_test_x
    return scaled_train_x


def generate_plotbars(inputData: pd.DataFrame):
    """
    Use for generate plotbars providing input tha performace dataframe.

    Parameters
    -------------------------
    inputData: pd.DataFrame,
        DataFrame that contains all performance.
        
    Returns
    -------------------------
    """

    column_to_filter = ["model_name", "loss", "accuracy", "AUROC", "AUPRC", "use_feature_selection", "run_type"]
    barplots(
                inputData[column_to_filter],
                groupby=["model_name", "use_feature_selection", "run_type"],
                orientation="horizontal",
                height=8,
                bar_width=0.2
            )
