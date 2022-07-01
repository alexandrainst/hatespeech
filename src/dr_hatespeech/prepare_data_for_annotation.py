"""Prepares the weakly supervised data for annotation in LabelStudio."""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from .load_data import load_weakly_supervised_data


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def prepare_data_for_annotation(config: DictConfig) -> None:
    """Prepares the weakly supervised data for annotation in LabelStudio."""

    # Load the weakly supervised data
    data_dict = load_weakly_supervised_data(config)
    df = data_dict["df"]
    data_path = data_dict["path"]

    # Use the weakly supervised labels to sample a balanced subset of the data
    sample_size = config.data.val_size + config.data.test_size
    df_off = df.query("label == 1").sample(sample_size // 2, random_state=config.seed)
    df_not = df.query("label == 0").sample(sample_size // 2, random_state=config.seed)

    # Concatenate the dataframes, shuffle them and extract the text
    df_merged = pd.concat([df_off, df_not]).sample(frac=1, random_state=config.seed)
    texts = df_merged.text.tolist()

    # Save the data to a txt file
    data_dir = Path(config.data.processed_dir)
    fname = data_path.stem.replace("_weakly_supervised", "_for_annotation") + ".txt"
    path = data_dir / fname
    path.write_text("\n".join(texts))


if __name__ == "__main__":
    prepare_data_for_annotation()
