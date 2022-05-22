'''Weak supervision module to create labels in an unsupervised setting.'''

from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from typing import Union
from pathlib import Path
from data import load_data
from labeling_functions import (
    contains_offensive_word,
    is_mention,
    is_dr_answer,
    use_guscode_model,
    use_danlp_electra_model,
    use_danlp_dabert_model,
    use_tfidf_model,
    has_been_moderated,
    sentiment
)



def main(data_dir: Union[str, Path] = "data"):
    '''Generate weakly supervised labels for the data.

    Args:
        data_dir (str or Path, optional):
            The path to the data directory. Defaults to 'data'.
    '''
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Create the path to the processed data directory
    processed_dir = data_dir / "processed"

    # Ensure that the processed data directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    df_train = load_data(data_dir=data_dir)

    # Define the list of labeling functions
    lfs = [
        contains_offensive_word,
        is_mention,
        is_dr_answer,
        use_guscode_model,
        use_danlp_electra_model,
        use_danlp_dabert_model,
        use_tfidf_model,
        has_been_moderated,
        sentiment
    ]

    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    train = applier.apply(df_train)

    # Train the label model
    label_model = LabelModel(cardinality=2)
    label_model.fit(train, n_epochs=500, log_freq=50, seed=4242)

    # Compute the training labels and add them to the dataframe
    df_train["label"] = label_model.predict(L=train, tie_break_policy="abstain")

    # Remove the abstained data points
    df_train = df_train[df_train.label != -1]

    # Save the dataframe
    path_str = [str(path) for path in processed_dir.glob('*_processed.parquet')
            if not path.name.startswith('test_')][0]
    path = Path(path_str.replace('_processed', '_weakly_supervised'))
    df_train.to_parquet(path)


if __name__ == "__main__":
    main()
