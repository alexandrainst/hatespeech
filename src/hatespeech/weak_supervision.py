"""Weak supervision module to create labels in an unsupervised setting."""

import logging
import multiprocessing as mp
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from snorkel.labeling.apply.dask import PandasParallelLFApplier
from snorkel.labeling.model import LabelModel

from . import labelling_functions as lfs
from .load_data import load_cleaned_data

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def apply_weak_supervision(config: DictConfig) -> pd.DataFrame:
    """Generate weakly supervised labels for the data.

    Args:
        config (DictConfig):
            The configuration.

    Returns:
        Pandas DataFrame:
            The data with weakly supervised labels.
    """
    # Load the cleaned data
    df = load_cleaned_data(config)

    # Log progress
    logger.info("Loading models to be used in the weak labelling")

    # Define the list of labelling functions
    lf_list = [
        lfs.contains_offensive_word,
        lfs.is_all_caps,
        lfs.contains_positive_swear_word,
        lfs.is_mention,
        lfs.is_dr_answer,
        lfs.use_danlp_model,
        lfs.use_attack_model,
        lfs.use_tfidf_model,
        lfs.has_been_moderated,
        lfs.has_positive_sentiment,
        lfs.is_spam,
    ]

    # Log progress
    logger.info(f"Applying weak supervision with {len(lf_list)} labelling functions")

    # Apply the LFs to the unlabeled training data
    n_jobs = mp.cpu_count() - 1 if config.n_jobs == -1 else config.n_jobs
    applier = PandasParallelLFApplier(lf_list)
    lf_df = applier.apply(df, n_parallel=n_jobs)

    # Train the label model
    label_model = LabelModel(cardinality=2)
    label_model.fit(lf_df, n_epochs=100, log_freq=50, seed=4242)

    # Compute the training labels and add them to the dataframe
    df["label"] = label_model.predict(L=lf_df, tie_break_policy="abstain")

    # Remove the abstained data points
    df = df.query("label != -1")

    # Save the dataframe
    data_dir = Path(config.data.weakly_supervised.dir)
    path = data_dir / config.data.weakly_supervised.fname
    df.to_parquet(path)

    # Return the dataframe
    return df


if __name__ == "__main__":
    apply_weak_supervision()
