"""Weak supervision module to create labels in an unsupervised setting."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from snorkel.labeling.model import LabelModel

from . import labelling_functions as lfs
from .load_data import load_cleaned_data
from .snorkel_utils import ImprovedPandasLFApplier

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
    df = load_cleaned_data(config).iloc[:100_000]

    # Load the models to be used in the labelling functions
    logger.info("Loading models to be used in the weak labelling")
    lfs.initialise_models()

    # Define the list of labelling functions
    lf_list = [
        lfs.contains_offensive_word,
        lfs.is_all_caps,
        lfs.contains_positive_swear_word,
        lfs.is_dr_answer,
        lfs.has_been_moderated,
        lfs.is_spam,
        lfs.is_mention,
        lfs.use_danlp_model,
        lfs.use_attack_model,
        lfs.use_tfidf_model,
        lfs.has_positive_sentiment,
    ]

    # Log progress
    logger.info(f"Applying weak supervision with {len(lf_list)} labelling functions")

    # Apply the LFs to the unlabeled training data
    applier = ImprovedPandasLFApplier(lf_list)
    lf_output_arr = applier.apply(df, batch_size=config.lf_batch_size)

    # Train the label model on the labelling function outputs
    label_model = LabelModel(cardinality=2)
    label_model.fit(lf_output_arr, n_epochs=100, log_freq=50, seed=4242)

    # Compute the training labels and add them to the dataframe
    df["label"] = label_model.predict(L=lf_output_arr, tie_break_policy="abstain")

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
