"""Train a simple TF-IDF + logistic regression model on the DKHate dataset."""

from pathlib import Path

import hydra
import joblib
from datasets import load_dataset
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.pipeline import Pipeline


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def train_tfidf_model(config: DictConfig) -> Pipeline:
    """Train a logitstic regression model with TF-IDF features on DKHate.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        Pipeline:
            The trained model.
    """
    # Get model config
    model_config = config.tfidf_model

    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        analyzer=model_config.analyzer,
        max_features=model_config.max_features,
        norm=None if model_config.norm == "None" else model_config.norm,
        lowercase=model_config.lowercase,
        ngram_range=model_config.ngram_range,
        min_df=model_config.min_df,
        max_df=model_config.max_df,
        smooth_idf=model_config.smooth_idf,
        sublinear_tf=model_config.sublinear_tf,
        use_idf=model_config.use_idf,
    )

    # Create the logistic regression model
    model = LogisticRegression(max_iter=model_config.max_iter)

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
    output_path = Path(config.models.dir) / model_config.fname
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    # Return the pipeline
    return pipeline


if __name__ == "__main__":
    train_tfidf_model()
