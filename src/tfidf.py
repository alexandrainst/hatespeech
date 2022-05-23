"""Train a simple TF-IDF + logistic regression model on the DKHate dataset."""

from datasets import load_dataset
from pathlib import Path
from typing import Union
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef


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
        ngram_range=(1, 3),
        min_df=0.01,
        max_df=0.9,
        max_features=10000,
        sublinear_tf=True,
    )

    # Create the logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Create the pipeline
    pipeline = Pipeline([("vectorizer", vectorizer), ("model", model)])

    # Load the DKHate dataset
    dataset_dict = load_dataset("DDSC/dkhate", use_auth_token=True)
    train_df = dataset_dict["train"].to_pandas()
    test_df = dataset_dict["test"].to_pandas()

    # Fit the vectorizer to the data and transform it
    pipeline.fit(train_df.text, train_df.label)

    # Score the model and output the results
    train_preds = pipeline.predict(train_df.text)
    test_preds = pipeline.predict(test_df.text)
    train_score = matthews_corrcoef(train_df.label, train_preds)
    test_score = matthews_corrcoef(test_df.label, test_preds)
    print(f"Train score: {train_score}")
    print(f"Test score: {test_score}")

    # Save the model
    joblib.dump(pipeline, output_path)

    # Save the pipeline
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)


if __name__ == "__main__":
    train_tfidf_model()
