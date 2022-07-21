"""Weak supervision module to create labels in an unsupervised setting."""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel

from . import labeling_functions as lfs
from .load_data import load_cleaned_data


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

    # Define the list of labeling functions
    lf_list = [
        lfs.contains_offensive_word,
        lfs.is_all_caps,
        lfs.contains_positive_swear_word,
        lfs.is_mention,
        lfs.is_dr_answer,
        lfs.use_hatespeech_model,
        lfs.use_tfidf_model,
        lfs.has_been_moderated,
        lfs.has_positive_sentiment,
    ]

    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lf_list)
    lf_df = applier.apply(df)

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
