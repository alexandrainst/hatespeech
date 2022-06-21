"""Weak supervision module to create labels in an unsupervised setting."""

from pathlib import Path

import hydra
from omegaconf import DictConfig
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel

from .labeling_functions import (
    contains_offensive_word,
    has_been_moderated,
    is_dr_answer,
    is_mention,
    sentiment,
    use_tfidf_model,
    use_transformer_ensemble,
)
from .load_data import load_cleaned_data


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def apply_weak_supervision(config: DictConfig):
    """Generate weakly supervised labels for the data.

    Args:
        config (DictConfig):
            The configuration.
    """
    # Load the cleaned data
    data_dict = load_cleaned_data(config=config)
    df_train = data_dict["df"]
    data_path = data_dict["path"]

    # Define the list of labeling functions
    lfs = [
        contains_offensive_word,
        is_mention,
        is_dr_answer,
        use_transformer_ensemble,
        use_tfidf_model,
        has_been_moderated,
        sentiment,
    ]

    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    train = applier.apply(df_train)

    # Train the label model
    label_model = LabelModel(cardinality=2)
    label_model.fit(train, n_epochs=100, log_freq=50, seed=4242)

    # Compute the training labels and add them to the dataframe
    df_train["label"] = label_model.predict(L=train, tie_break_policy="abstain")

    # Remove the abstained data points
    df_train = df_train[df_train.label != -1]

    # Save the dataframe
    fname = str(data_path.name).replace("_cleaned", "_weakly_supervised")
    path = Path(config.final.dir) / fname
    df_train.to_parquet(path)


if __name__ == "__main__":
    apply_weak_supervision()
