"""Train a simple TF-IDF + logistic regression model on the DKHate dataset."""

from pathlib import Path
from typing import Union

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.pipeline import Pipeline


def train_tfidf_model(output_path: Union[str, Path] = "models/tfidf_model.bin"):
    """Train a logitstic regression model with TF-IDF features on DKHate.

    This will save the model to `output_path`.

    Args:
        output_path (str or Path, optional):
            The path to save the model to. Defaults to
            'models/tfidf_model.bin'.
    """
    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        analyzer="word",
        max_features=10_000,
        norm=None,
    )

    # Create the logistic regression model
    model = LogisticRegression(max_iter=10_000)

    # Create the pipeline
    pipeline = Pipeline([("vectorizer", vectorizer), ("model", model)])

    # Load the DKHate dataset
    dataset_dict = load_dataset("DDSC/dkhate", use_auth_token=True)
    train_df = dataset_dict["train"].to_pandas()
    test_df = dataset_dict["test"].to_pandas()

    # Convert labels to 0/1
    train_df["label"] = train_df["label"].apply(lambda x: 1 if x == "OFF" else 0)
    test_df["label"] = test_df["label"].apply(lambda x: 1 if x == "OFF" else 0)

    # Fit the vectorizer to the data and transform it
    pipeline.fit(train_df.text, train_df.label)

    # Score the model and output the results
    train_preds = pipeline.predict(train_df.text)
    test_preds = pipeline.predict(test_df.text)
    train_mcc = matthews_corrcoef(train_df.label, train_preds)
    train_f1 = f1_score(train_df.label, train_preds)
    test_mcc = matthews_corrcoef(test_df.label, test_preds)
    test_f1 = f1_score(test_df.label, test_preds)
    print(f"Train MCC: {100 * train_mcc:.2f}%")
    print(f"Train F1: {100 * train_f1:.2f}%")
    print(f"Test MCC: {100* test_mcc:.2f}%")
    print(f"Test F1: {100 * test_f1:.2f}%")

    # Save the pipeline
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)


if __name__ == "__main__":
    train_tfidf_model()
