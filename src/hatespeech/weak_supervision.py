"""Weak supervision module to create labels in an unsupervised setting."""

import logging
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from snorkel.labeling import LabelingFunction
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
    df = load_cleaned_data(config)

    # Load the models to be used in the labelling functions
    logger.info("Loading models to be used in the weak labelling")
    lfs.initialise_models()

    # Define mapping from config options to their associated labelling functions
    params = dict(
        filter_dr_answer=not config.label_model.ignore_is_dr_answer,
        filter_spam=not config.label_model.ignore_is_spam,
    )
    lf_mapping = dict(
        ignore_contains_offensive_word=LabelingFunction(
            name="contains_offensive_word",
            f=partial(lfs.contains_offensive_word, **params),
        ),
        ignore_is_all_caps=LabelingFunction(name="is_all_caps", f=lfs.is_all_caps),
        ignore_contains_positive_swear_word=LabelingFunction(
            name="contains_positive_swear_word",
            f=lfs.contains_positive_swear_word,
        ),
        ignore_is_dr_answer=LabelingFunction(name="is_dr_answer", f=lfs.is_dr_answer),
        ignore_has_been_moderated=LabelingFunction(
            name="has_been_moderated", f=partial(lfs.has_been_moderated, **params)
        ),
        ignore_is_spam=LabelingFunction(name="is_spam", f=lfs.is_spam),
        ignore_is_mention=LabelingFunction(
            name="is_mention", f=partial(lfs.is_mention, **params)
        ),
        ignore_use_danlp_model=LabelingFunction(
            name="use_danlp_model", f=partial(lfs.use_danlp_model, **params)
        ),
        ignore_use_attack_model=LabelingFunction(
            name="use_attack_model", f=partial(lfs.use_attack_model, **params)
        ),
        ignore_use_tfidf_model=LabelingFunction(
            name="use_tfidf_model", f=partial(lfs.use_tfidf_model, **params)
        ),
        ignore_has_positive_sentiment=LabelingFunction(
            name="has_positive_sentiment",
            f=lfs.has_positive_sentiment,
        ),
    )

    # Define the list of labelling functions
    lf_list = [
        lf for ignore_lf, lf in lf_mapping.items() if not config.label_model[ignore_lf]
    ]

    # Log progress
    logger.info(f"Applying weak supervision with {len(lf_list)} labelling functions")

    # Apply the LFs to the unlabeled training data
    applier = ImprovedPandasLFApplier(lf_list)
    lf_output_arr = applier.apply(df, batch_size=config.label_model.lf_batch_size)

    # Train the label model on the labelling function outputs
    label_model = LabelModel(cardinality=2)
    label_model.fit(
        lf_output_arr,
        n_epochs=config.label_model.epochs,
        log_freq=config.label_model.log_freq,
        seed=config.seed,
    )

    # Compute the training labels and add them to the dataframe
    df["label"] = label_model.predict(L=lf_output_arr, tie_break_policy="abstain")

    # Remove the abstained data points
    df = df.query("label != -1")

    # Save the dataframe
    data_dir = Path(config.data.weakly_supervised.dir)
    fname = config.data.weakly_supervised.fname.format(config.label_model.name)
    df.to_parquet(data_dir / fname)

    # Return the dataframe
    return df


if __name__ == "__main__":
    apply_weak_supervision()
