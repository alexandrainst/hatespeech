"""Labeling functions used for weak supervision."""

import re
import warnings

import joblib
import nltk
import torch
from snorkel.labeling import labeling_function
from transformers.pipelines import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

# Download word tokenizer
nltk.download("punkt")


# Create label names
ABSTAIN = -1
NOT_OFFENSIVE = 0
OFFENSIVE = 1


# Get device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# Load TF-IDF hatespeech model
tfidf = joblib.load("models/tfidf_model.bin")


# Load NER model
ner_model_id = "saattrupdan/nbailab-base-ner-scandi"
ner_tok = AutoTokenizer.from_pretrained(ner_model_id, cache_dir=".cache")
ner_model = (
    AutoModelForTokenClassification.from_pretrained(ner_model_id, cache_dir=".cache")
    .eval()
    .to(device)
)


# Load transformer hatespeech models
hatespeech_model_id = "DaNLP/da-electra-hatespeech-detection"
hatespeech_tok = AutoTokenizer.from_pretrained(hatespeech_model_id, cache_dir=".cache")
hatespeech_model = (
    AutoModelForSequenceClassification.from_pretrained(
        hatespeech_model_id, cache_dir=".cache"
    )
    .eval()
    .to(device)
)


# Load sentiment model
sent_model_id = "pin/senda"
sent_tok = AutoTokenizer.from_pretrained(sent_model_id, cache_dir=".cache")
sent_model = (
    AutoModelForSequenceClassification.from_pretrained(
        sent_model_id, cache_dir=".cache"
    )
    .eval()
    .to(device)
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
            The assigned label, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Extract the document
    doc = record.text

    # Define list of offensive words
    offensive_words = [
        r"pe+rke+r(\W|$)",
        r"[sz]no+tu+nge",
        r"pi+k(\W|$)",
        r"fi+[sz]+e",
        r"re+ta+r?deret",
        r"a+[sz]+ho+le+",
        r"re+ta+rd",
        r"i+di+o+t",
        r"ho+ld (do+g |nu+ |ba+re )?k[æ?]+ft",
        r"[sz]to+dder",
        r"mo+ngo+l",
        r"(m[ø?]+[gj]|kla+mm[eo]|u+[sz]+e+lt?) ?(dy+r|[sz]vi+n|[sz]o+|ko+[sz]t)",
        r"kla+pha+t",
        r"(?<!flue)kne+ppe+",
        r"li+de+rli+g",
        r"va+tpi+k",
        r"k[æ?]+ll*i+ng",
        r"fa+t[sz]va+g",
        r"gi+mpe",
        r"(\W|^)lu+der",
        r"du+mb +fu+ck",
        r"a+f[sz]ku+m",
        r"p[sz]y+ko+pa+t",
        r"b[ø?]+[sz]+er[ø?]+v",
        r"(\W|^)tu+mpe",
        r"[åa?]+nd[sz]+va+g",
        r"[sz]pa+[sz]+e+r",
        r"r[ø?]+vhu+l",
        r"t[åa?]+be",
        r"pe+rve+r[sz]",
        r"[sz]to+dd*e+r",
        r"t[åa?]+ge+[sz]na+k",
        r"(din|den|nogle|nogen) ke+gle+r?",
        r"[sz][ck]u+m ?ba+g",
        r"pu+tin ?fa+n",
        r"fjo+l[sz]",
        r"h[æ?]+klefejl i+ ky+[sz]en",
        r"(\W|^)[åa?]+nd[sz]+va+g",
        r"(\W|^)e+le+ndi+g",
        r"(\W|^)([sz]va+g)?pi+[sz]+(e|er|ere)?(\W|$)",
    ]

    # Mark document as offensive if it contains an offensive word, and abstain
    # otherwise
    if is_dr_answer(record) == NOT_OFFENSIVE:
        return NOT_OFFENSIVE
    elif any(re.search(regex, doc.lower()) for regex in offensive_words):
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def is_all_caps(record) -> int:
    """Check if the document is written in all caps.

    This will mark the document as offensive if it is written in all caps , and abstain
    otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            The assigned label, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Extract the document
    doc = record.text

    # Mark document as offensive if it is written in all caps, and abstain otherwise
    if doc.isupper():
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def contains_positive_swear_word(record) -> int:
    """Check if the document contains a swear word used in a positive way.

    This will mark the document as not offensive if it contains a swear word used in a
    positive way, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            The assigned label, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Extract the document
    doc = record.text

    # Define list of offensive words
    positive_swear_words = [
        r"(\W|^)(fa+nde?me|fu+ck|fu+cki+ng|k[æ?]+ft|[sz]ate?me) "
        r"(hvo+r )?(e+r)?((de+t|du+|de+|ha+n|hu+n))?"
        r"?(vi+ld|ja|fe+d|go+d?|l[æ?]+kk*e+r|ni+ce|[sz]jo+v|[sz]e+j)",
        r"^f+u+c+k+( m+a+n+)? *[?!.]* *$",
        r"ho+ld (da+ )?k[æ?]+ft",
    ]

    # Mark document as not offensive if it contains a positive swear word, and abstain
    # otherwise
    if any(re.search(regex, doc.lower()) for regex in positive_swear_words):
        return NOT_OFFENSIVE
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
            The assigned label, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Extract the document
    doc = record.text

    # Split up the document into words
    words = nltk.word_tokenize(doc)

    # Set `model_max_length` if not specified
    if ner_tok.model_max_length > 100_000:
        ner_tok.model_max_length = 512

    # Tokenise the words
    inputs = ner_tok(
        words, truncation=True, return_tensors="pt", is_split_into_words=True
    )

    # Get the list of word indices
    word_idxs = inputs.word_ids()

    # Move the tokens to the desired device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the model predictions
    with torch.no_grad():
        predictions = ner_model(**inputs).logits[0]

    # Extract the NER tags
    ner_tags = [
        ner_model.config.id2label[label_id.item()]
        for label_id in predictions.argmax(dim=-1)
    ]

    # Propagate the NER tags from the first token in each word to the rest of the word
    ner_tag = "O"
    for idx in range(1, len(ner_tags)):
        beginning_of_word = word_idxs[idx - 1] != word_idxs[idx]
        if beginning_of_word:
            ner_tag = ner_tags[idx]
        else:
            ner_tags[idx] = ner_tag

    # Remove the first and last token from the list of NER tags, as they are just
    # the special tokens <s> and </s>
    ner_tags = ner_tags[1:-1]

    # Count all the non-person tokens
    num_non_person_tokens = sum(1 for tag in ner_tags if not tag.endswith("PER"))

    # If all the tokens are person tokens then mark the document as not offensive, and
    # abstain otherwise
    if num_non_person_tokens == 0:
        return NOT_OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def is_dr_answer(record) -> int:
    """Check if the document is an official reply from DR.

    This will mark the document as not offensive if it contains an official reply from
    DR, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            The assigned label, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
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
def use_hatespeech_model(record) -> int:
    """Apply an ensemble of hatespeech detection transformer models.

    This will apply the model DaNLP/Electra-hatespeech-detection.

    This will mark the document as offensive if the model predicts the document as
    offensive with confidence above 70%, as not offensive if the model predicts the
    document as not offensive with confidence above 99.9%, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            The assigned label, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Extract the document
    doc = record.text

    # Set `model_max_length` if not specified
    if hatespeech_tok.model_max_length > 100_000:
        hatespeech_tok.model_max_length = 512

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # Tokenise the document
        inputs = hatespeech_tok(doc, truncation=True, return_tensors="pt")

        # Move the tokens to the desired device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the predictions
        pred = hatespeech_model(**inputs).logits[0]

        # Extract the offensive probability
        offensive_prob = torch.softmax(pred, dim=-1)[-1].item()

    # If the model predicts that the document is offensive with confidence above 50%
    # then mark it as offensive, if it predicts it is not offensive with confidence
    # above 99.9% then mark it as not offensive, otherwise abstain
    if is_dr_answer(record) == NOT_OFFENSIVE or offensive_prob < 0.001:
        return NOT_OFFENSIVE
    elif offensive_prob > 0.5:
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def use_tfidf_model(record) -> int:
    """Apply the TF-IDF offensive speech detection model.

    This will mark the document as offensive if the model classifies it as offensive
    with a decision score greater than 2, and abstains otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            The assigned label, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Extract the document
    doc = record.text

    # Get the prediction score
    predicted_score = tfidf.decision_function([doc])[0]

    # If the predictive score is positive then mark as offensive, and otherwise abstain
    if is_dr_answer(record) == NOT_OFFENSIVE:
        return NOT_OFFENSIVE
    elif predicted_score > 2:
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def has_been_moderated(record) -> int:
    """Check if a document has already been moderated.

    This will mark the document as offensive if it has been moderated, and abstain
    otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document has been moderated, and -1
            (abstain) otherwise.
    """
    # Extract the moderation action
    action = record.action

    # If the action is not "none" then mark the document as offensive, and otherwise
    # abstain
    if is_dr_answer(record) == NOT_OFFENSIVE:
        return NOT_OFFENSIVE
    elif action != "none":
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def has_positive_sentiment(record) -> int:
    """Apply a sentiment analysis model.

    This will mark the document as not offensive if the probability of the document
    being negative is less than 10%, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 0 (not offensive) if the document is classified as not
            offensive by the model, and -1 (abstain) otherwise.
    """
    # Extract the document
    doc = record.text

    # Set `model_max_length` if not specified
    if sent_tok.model_max_length > 100_000:
        sent_tok.model_max_length = 512

    # Get the prediction
    with warnings.catch_warnings():
        with torch.no_grad():
            warnings.simplefilter("ignore", category=UserWarning)

            # Tokenise the document
            inputs = sent_tok(doc, truncation=True, return_tensors="pt")

            # Move the tokens to the desired device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get the prediction
            prediction = sent_model(**inputs).logits[0]

            # Extract the probability of the document being negative
            negative_prob = torch.softmax(prediction, dim=-1)[0].item()

    # If the probability of the document being negative is below 30% then mark it as
    # not offensive, and otherwise abstain
    if negative_prob < 0.1:
        return NOT_OFFENSIVE
    else:
        return ABSTAIN
