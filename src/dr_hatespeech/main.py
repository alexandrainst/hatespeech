"""Main script which processes data and trains models."""

import hydra
from omegaconf import DictConfig

from .train_transformer import train_transformer_model


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    """Main script which processes data and trains models.

    Args:
        config (DictConfig):
            Configuration object.
    """
    # Try to load the final data
    try:
        train_transformer_model(config)

    # Otherwise there are no splits available, so we split the data
    except FileNotFoundError:
        try:
            from .split_data import split_data

            split_data(config)
            train_transformer_model(config)

        # Otherwise there is no weakly supervised data available, so we try to perform
        # the weak supervision
        except FileNotFoundError:
            try:
                from .split_data import split_data
                from .weak_supervision import apply_weak_supervision

                apply_weak_supervision(config)
                split_data(config)
                train_transformer_model(config)

            # Otherwise either there is no cleaned data available and/or there is no
            # trained TF-IDF model, so we clean the data and/or train the TF-IDF model
            except FileNotFoundError as e:
                from .clean_data import clean_data
                from .split_data import split_data
                from .train_tfidf import train_tfidf_model
                from .weak_supervision import apply_weak_supervision

                if "cleaned" in str(e):
                    clean_data(config)
                if "tfidf" in str(e):
                    train_tfidf_model(config)
                apply_weak_supervision(config)
                split_data(config)
                train_transformer_model(config)


if __name__ == "__main__":
    main()
