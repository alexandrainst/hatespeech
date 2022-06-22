"""Labeling functions used for weak supervision."""

from snorkel.labeling import labeling_function
from transformers.pipelines import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import re
import joblib
import torch
import warnings


# Create label names
ABSTAIN = -1
NOT_OFFENSIVE = 0
OFFENSIVE = 1


# Set up parameters for `transformers` pipelines
pipe_params = dict(truncation=True, max_length=512)


# Get device
device = 0 if torch.cuda.is_available() else -1


# Load NER model
ner = pipeline(model="saattrupdan/nbailab-base-ner-scandi", device=device)

# Load transformer hatespeech models
hatespeech_model_ids = [
    "DaNLP/da-electra-hatespeech-detection",
    "DaNLP/da-bert-hatespeech-detection",
    "DaNLP/da-electra-hatespeech-detection",
]
hatespeech_toks = [
    AutoTokenizer.from_pretrained(model_id) for model_id in hatespeech_model_ids
]
hatespeech_models = [
    AutoModelForSequenceClassification.from_pretrained(model_id)
    .eval()
    .to("cuda" if device == 0 else "cpu")
    for model_id in hatespeech_model_ids
]

# Load TF-IDF hatespeech model
tfidf = joblib.load("models/tfidf_model.bin")

# Load sentiment model
sent_model_id = "pin/senda"
sent_tok = AutoTokenizer.from_pretrained(sent_model_id)
sent_model = (
    AutoModelForSequenceClassification.from_pretrained(sent_model_id)
    .eval()
    .to("cuda" if device == 0 else "cpu")
)


@labeling_function()
def contains_offensive_word(record) -> int:
    """Check if the document contains an offensive word.

    This will mark the document as offensive if it contains an offensive word
    and is not an official DR answer, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document contains an offensive
            word, and -1 (abstain) otherwise.
    """
    # Extract the document
    doc = record.text

    # Define list of offensive words
    offensive_words = [
        r"perker",
        r"snotunge",
        r"svagpisser",
        r"pik\W",
        r"fisse",
        r"retar?deret",
        r"asshole",
        r"retard",
        r"idiot",
        r"hold (dog |nu |bare )?k[æ?]ft",
        r"stodder",
        r"mongol",
        r"(m[ø?][gj]|klamme|usselt?) ?(svin|so|kost)",
        r"klaphat",
        r"(?<!flue)kneppe",
        r"liderlig",
        r"vatpik",
        r"k[æ?]lling",
        r"fatsvag",
        r"gimpe",
        r"\Wluder",
        r"dumb +fuck",
        r"afskum",
        r"psykopat",
        r"b[ø?]sser[ø?]v",
        r"\Wtumpe",
        r"[åa?]ndss?vag",
        r"spasser",
        r"r[ø?]vhul",
    ]

    # Mark document as offensive if it contains an offensive word, and abstain
    # otherwise
    if is_dr_answer(record) == ABSTAIN and any(
        re.search(regex, doc.lower()) for regex in offensive_words
    ):
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def is_mention(record) -> int:
    """Check if the document consists of only mentions.

    This will mark the document as not offensive if it consists of only mentions, and
    abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 0 (not offensive) if the document consists of only mentions,
            and -1 (abstain) otherwise.
    """
    # Extract the document
    doc = record.text

    # Get the PER label indices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        per_indices = [
            (dct["start"], dct["end"])
            for dct in ner(doc, aggregation_strategy="first")
            if "entity_group" in dct and dct["entity_group"] == "PER"
        ]

    # Sort the indices, so that the latest mention is first
    per_indices = sorted(per_indices, key=lambda x: x[0], reverse=True)

    # Remove all PER labels, by removing them one at a time, starting from the
    # end of the document
    for start, end in per_indices:
        doc = doc[:start] + doc[end:]

    # Strip the document of whitespace
    doc = doc.strip()

    # If the remaining document is empty then mark it as not offensive,
    # otherwise abstain
    if doc == "":
        return NOT_OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def is_dr_answer(record) -> int:
    """Check if the document is an official reply from DR.

    This will mark the document as not offensive if it contains an official
    reply from DR, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 0 (not offensive) if the document contains an
            official reply from DR, and -1 (abstain) otherwise.
    """
    # Extract the document
    doc = record.text

    # Define phrases the DR employees tend to use
    dr_phrases = [
        r"vi har slettet din kommentar",
        r"overskrider vores retningslinjer",
        r"\W(m?vh\.?|/+) *[a-zæøå]+ *[/,]+ *dr",
    ]

    # If the document contains any of the phrases then mark it as not
    # offensive, otherwise abstain
    if any(re.search(regex, doc.lower()) for regex in dr_phrases):
        return NOT_OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def use_transformer_ensemble(record) -> int:
    """Apply an ensemble of hatespeech detection transformer models.

    The following models will be applied:

        - DaNLP/Electra-hatespeech-detection
        - Guscode/DKbert-hatespeech-detection
        - DaNLP/da-bert-hatespeech-detection

    This will mark the document as offensive if all models predict the document
    as offensive with confidence above 70%, as not offensive if all models
    predict the document as not offensive with confidence above 99.9%, and
    abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document is classified as
            offensive by the ensemble, and 0 (not offensive) if the document is
            classified as not offensive by the ensemble.
    """
    # Extract the document
    doc = record.text

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # Tokenise the document
        tokenised = [
            tok(doc, **pipe_params, return_tensors="pt") for tok in hatespeech_toks
        ]

        # Move the tokens to the desired device
        tokenised = [
            {k: v.to("cuda" if device == 0 else "cpu") for k, v in dct.items()}
            for dct in tokenised
        ]

        # Get the predictions
        preds = [
            model(**tokens).logits[0]
            for tokens, model in zip(tokenised, hatespeech_models)
        ]

        # Extract the offensive probability
        offensive_probs = [torch.softmax(pred, dim=-1)[-1].item() for pred in preds]

    # If all the models predict that the document is offensive with confidence
    # above 70% then mark it as offensive, if they all predict it is not
    # offensive with confidence above 99.9% then mark it as not offensive,
    # otherwise abstain
    if all(prob > 0.7 for prob in offensive_probs):
        return OFFENSIVE
    elif all(prob < 0.001 for prob in offensive_probs):
        return NOT_OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def use_tfidf_model(record) -> int:
    """Apply the TF-IDF offensive speech detection model.

    This will mark the document as offensive if the model classifies it as
    offensive with a positive decision score, and abstains otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document is classified as
            offensive by the model, and 0 (not offensive) if the document is
            classified as not offensive by the model.
    """
    # Extract the document
    doc = record.text

    # Get the prediction score
    predicted_score = tfidf.decision_function([doc])[0]

    # If the predictive score is positive then mark as offensive, and otherwise
    # abstain
    if predicted_score > 0:
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def has_been_moderated(record) -> int:
    """Check if a document has already been moderated.

    This will mark the document as offensive if it has been moderated, and
    abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document has been moderated,
            and -1 (abstain) otherwise.
    """
    # Extract the moderation action
    action = record.action

    # If the action is not "none" then mark the document as offensive, and
    # otherwise abstain
    if action != "none":
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def sentiment(record) -> int:
    """Apply a sentiment analysis model.

    This will mark the document as not offensive if the probability of the
    document being negative is less than 30%, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 0 (not offensive) if the document is classified as
            not offensive by the model, and -1 (abstain) otherwise.
    """
    # Extract the document
    doc = record.text

    # Get the prediction
    with warnings.catch_warnings():
        with torch.no_grad():
            warnings.simplefilter("ignore", category=UserWarning)

            # Tokenise the document
            inputs = sent_tok(doc, **pipe_params, return_tensors="pt")

            # Move the tokens to the desired device
            inputs = {
                k: v.to("cuda" if device == 0 else "cpu") for k, v in inputs.items()
            }

            # Get the prediction
            prediction = sent_model(**inputs).logits[0]

            # Extract the probability of the document being negative
            negative_prob = torch.softmax(prediction, dim=-1)[0].item()

    # If the probability of the document being negative is below 30% then mark
    # it as not offensive, and otherwise abstain
    if negative_prob < 0.3:
        return NOT_OFFENSIVE
    else:
        return ABSTAIN
