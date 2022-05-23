"""Unit tests related to the tfidf module."""

from pathlib import Path
from src.tfidf import train_tfidf_model


def test_train_tfidf_model():
    train_tfidf_model()
    model_path = Path("models") / "tfidf_model.bin"
    assert model_path.exists()
