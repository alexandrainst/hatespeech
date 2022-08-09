"""Labeling functions used for weak supervision."""

import re
import warnings
from typing import Tuple

import joblib
import nltk
import numpy as np
import torch
from snorkel.labeling import labeling_function
from tqdm.auto import tqdm
from transformers.pipelines import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from .attack import load_attack

# Create label names
ABSTAIN = -1
NOT_OFFENSIVE = 0
OFFENSIVE = 1


# Get device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def initialise_models():
    """Initialise the models used in the weak supervision."""

    # Set global variables
    global tfidf
    global ner_tok, ner_model
    global danlp_tok, danlp_model
    global attack_tok, attack_model
    global sent_tok, sent_model

    # Initialise progress bar
    with tqdm(desc="Loading models", total=6, leave=False) as pbar:

        # Download word tokenizer
        nltk.download("punkt", quiet=True)
        pbar.update()

        # Load TF-IDF hatespeech model
        tfidf = joblib.load("models/tfidf_model.bin")
        pbar.update()

        # Load NER model
        ner_model_id = "saattrupdan/nbailab-base-ner-scandi"
        ner_tok = AutoTokenizer.from_pretrained(ner_model_id, cache_dir=".cache")
        ner_model = (
            AutoModelForTokenClassification.from_pretrained(
                ner_model_id, cache_dir=".cache"
            )
            .eval()
            .to(DEVICE)
        )
        pbar.update()

        # Load DaNLP hatespeech model
        danlp_model_id = "DaNLP/da-electra-hatespeech-detection"
        danlp_tok = AutoTokenizer.from_pretrained(danlp_model_id, cache_dir=".cache")
        danlp_model = (
            AutoModelForSequenceClassification.from_pretrained(
                danlp_model_id, cache_dir=".cache"
            )
            .eval()
            .to(DEVICE)
        )
        pbar.update()

        # Load A-ttack hatespeech model
        attack_tok, attack_model = load_attack()
        attack_model.eval().to(DEVICE)
        pbar.update()

        # Load sentiment model
        sent_model_id = "pin/senda"
        sent_tok = AutoTokenizer.from_pretrained(sent_model_id, cache_dir=".cache")
        sent_model = (
            AutoModelForSequenceClassification.from_pretrained(
                sent_model_id, cache_dir=".cache"
            )
            .eval()
            .to(DEVICE)
        )
        pbar.update()


@labeling_function()
def is_spam(record) -> np.ndarray:
    """Check if the document is spam.

    This will mark the document as not offensive if it is spam and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Extract the documents
    docs = record.text

    # Define list of spam phrases
    spam_phrases = [
        r"[Jj]eg sendte (dig)? en (venne|venskabs)?anmodning",
        r"[Jj]eg s[aå?] (lige )?din profil",
        r"sende dig en (venne|venskabs)?anmodning",
        r"kennenlernen",
    ]

    # Compute the final labels
    def compute_label(doc: str):
        if any(re.search(regex, doc.lower()) for regex in spam_phrases):
            return NOT_OFFENSIVE
        else:
            return ABSTAIN

    labels[:] = [compute_label(doc) for doc in docs]

    return labels


@labeling_function()
def contains_offensive_word(record) -> np.ndarray:
    """Check if the document contains an offensive word.

    This will mark the document as offensive if it contains an offensive word
    and is not an official DR answer, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array of the same shape as the input:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Check if any of the documents are DR answers or spam, and mark the remaining
    # documents that needs to be checked
    labels = np.maximum(labels, is_dr_answer(record))
    labels = np.maximum(labels, is_spam(record))

    # Extract the documents
    docs = record.iloc[
        [idx for idx, lbl in enumerate(labels) if lbl == ABSTAIN]
    ].text.tolist()

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
        r"ho+ld (di+n |do+g |nu+ |ba+re )k[æ?]+ft",
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

    # Compute the final labels
    def compute_label(doc: str):
        if any(re.search(regex, doc.lower()) for regex in offensive_words):
            return OFFENSIVE
        else:
            return ABSTAIN

    doc_idx = 0
    for idx in range(len(record)):
        if labels[idx] == ABSTAIN:
            labels[idx] = compute_label(docs[doc_idx])
            doc_idx += 1

    return labels


@labeling_function()
def is_all_caps(record) -> np.ndarray:
    """Check if the document is written in all caps.

    This will mark the document as offensive if it is written in all caps , and abstain
    otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Extract the documents
    docs = record.text

    # Compute the final labels
    def compute_label(doc: str):
        if doc.isupper():
            return OFFENSIVE
        else:
            return ABSTAIN

    labels[:] = [compute_label(doc) for doc in docs]

    return labels


@labeling_function()
def contains_positive_swear_word(record) -> np.ndarray:
    """Check if the document contains a swear word used in a positive way.

    This will mark the document as not offensive if it contains a swear word used in a
    positive way, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Extract the documents
    docs = record.text

    # Define list of offensive words
    positive_swear_words = [
        r"(\W|^)(fa+nde?me|fu+ck|fu+cki+ng|k[æ?]+ft|[sz]ate?me) "
        r"(hvo+r )?(e+r)?(de+t|du+|de+|ha+n|hu+n)?"
        r"?(vi+ld|ja|fe+d|go+d?|l[æ?]+kk*e+r|ni+ce|[sz]jo+v|[sz]e+j)",
        r"^f+u+c+k+( m+a+n+d?| j+a+)? *[?!.]* *$",
        r"f+u+c+k+( m+a+n+d?| j+a+) *[?!.]* *$",
        r"ho+ld da+ k[æ?]+ft",
    ]

    # Compute the final labels
    def compute_label(doc: str):
        if any(re.search(regex, doc.lower()) for regex in positive_swear_words):
            return NOT_OFFENSIVE
        else:
            return ABSTAIN

    labels[:] = [compute_label(doc) for doc in docs]

    return labels


@labeling_function()
def is_mention(record) -> np.ndarray:
    """Check if the document consists of only mentions.

    This will mark the document as not offensive if it consists of only mentions, and
    abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Load model if it has not been loaded yet
    if "ner_tok" not in globals() or "ner_model" not in globals():
        initialise_models()

    # Set `model_max_length` if not specified
    if ner_tok.model_max_length > 100_000:  # type: ignore [name-defined]
        ner_tok.model_max_length = 512  # type: ignore [name-defined]

    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Check if any of the documents are DR answers or spam, and mark the remaining
    # documents that needs to be checked
    labels = np.maximum(labels, is_dr_answer(record))
    labels = np.maximum(labels, is_spam(record))

    # Extract the documents
    docs = record.iloc[
        [idx for idx, lbl in enumerate(labels) if lbl == ABSTAIN]
    ].text.tolist()

    # Remove special characters, such as links, by removing all upper case letters
    # enclosed in square brackets
    docs = [re.sub(r"\[[A-ZÆØÅ]+\]", "", doc) for doc in docs]

    # Only preserve characters and spaces
    docs = [re.sub(r"[^A-ZÆØÅa-zæøå ]", "", doc) for doc in docs]

    # Remove duplicate and trailing spaces
    docs = [re.sub(r" +", " ", doc).strip() for doc in docs]

    # Split up the document into words
    try:
        words = [nltk.word_tokenize(doc) for doc in docs]
    except LookupError:
        initialise_models()
        words = [nltk.word_tokenize(doc) for doc in docs]

    # Tokenise the words
    inputs = ner_tok(  # type: ignore [name-defined]
        words,
        truncation=True,
        padding=True,
        return_tensors="pt",
        is_split_into_words=True,
    )

    # Get the list of word indices
    word_idxs = np.asarray([inputs.word_ids(idx) for idx in range(len(docs))])

    # Move the tokens to the desired device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Get the model predictions
    with torch.no_grad():
        predictions = ner_model(**inputs).logits.argmax(dim=-1)  # type: ignore [name-defined]

    # Extract the NER tags
    pad_idx = sent_tok.pad_token_id  # type: ignore [name-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ner_tags_list = [
            [
                ner_model.config.id2label[label_id.item()]  # type: ignore [name-defined]
                for label_id in label_tensor[label_tensor != pad_idx]
            ]
            for label_tensor in predictions
        ]

    # Propagate the NER tags from the first token in each word to the rest of the word
    for doc_idx, ner_tags in enumerate(ner_tags_list):
        ner_tag = "O"
        for token_idx in range(1, len(ner_tags)):
            beginning_of_word = (
                word_idxs[doc_idx, token_idx - 1] != word_idxs[doc_idx, token_idx]
            )
            if beginning_of_word:
                ner_tag = ner_tags_list[doc_idx][token_idx]
            else:
                ner_tags_list[doc_idx][token_idx] = ner_tag

    # Remove the first and last token from the list of NER tags, as they are just
    # the special tokens <s> and </s>
    ner_tags_list = [ner_tags[1:-1] for ner_tags in ner_tags_list]

    # Count all the non-person tokens
    num_non_person_tokens_list = [
        sum(1 for tag in ner_tags if not tag.endswith("PER"))
        for ner_tags in ner_tags_list
    ]

    # Get the documents with offensive words
    offensive_words = contains_offensive_word(record)

    # Zip up the counts and the offensive word labels
    pairs = list(zip(num_non_person_tokens_list, offensive_words))

    # Compute the final labels
    def compute_label(pair: Tuple[int, int]):
        num_non_person_tokens, offensive_word = pair
        contains_single_non_offensive_word = (
            num_non_person_tokens == 1 and offensive_word != OFFENSIVE
        )
        if num_non_person_tokens == 0 or contains_single_non_offensive_word:
            return NOT_OFFENSIVE
        else:
            return ABSTAIN

    pair_idx = 0
    for idx in range(labels.shape[0]):
        if labels[idx] == ABSTAIN:
            labels[idx] = compute_label(pairs[pair_idx])
            pair_idx += 1

    return labels


@labeling_function()
def is_dr_answer(record) -> np.ndarray:
    """Check if the document is an official reply from DR.

    This will mark the document as not offensive if it contains an official reply from
    DR, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Extract the documents
    docs = record.text

    # Define phrases the DR employees tend to use
    dr_phrases = [
        r"vi har slettet din kommentar",
        r"overskrider vores retningslinjer",
        r"\W(m?vh\.?|/+) *[a-zæøå]+ *[/,]+ *dr",
    ]

    # Compute the final labels
    def compute_label(doc: str):
        if any(re.search(regex, doc.lower()) for regex in dr_phrases):
            return NOT_OFFENSIVE
        else:
            return ABSTAIN

    labels[:] = [compute_label(doc) for doc in docs]

    return labels


@labeling_function()
def use_danlp_model(record) -> np.ndarray:
    """Apply the DaNLP ELECTRA hatespeech detection transformer model.

    This will apply the model DaNLP/da-electra-hatespeech-detection.

    This will mark the document as offensive if the model predicts the document as
    offensive with confidence above 50%, as not offensive if the model predicts the
    document as not offensive with confidence above 99.9%, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array of the same shape as the input:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Load model if it has not been loaded yet
    if "danlp_tok" not in globals() or "danlp_model" not in globals():
        initialise_models()

    # Set `model_max_length` if not specified
    if danlp_tok.model_max_length > 100_000:  # type: ignore [name-defined]
        danlp_tok.model_max_length = 512  # type: ignore [name-defined]

    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Check if any of the documents are DR answers or spam, and mark the remaining
    # documents that needs to be checked
    labels = np.maximum(labels, is_dr_answer(record))
    labels = np.maximum(labels, is_spam(record))
    labels = np.maximum(labels, is_mention(record))

    # Extract the documents
    docs = record.iloc[[idx for idx, lbl in enumerate(labels) if lbl == ABSTAIN]].text

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # Tokenise the document
        inputs = danlp_tok(  # type: ignore [name-defined]
            docs.tolist(), truncation=True, padding=True, return_tensors="pt"
        )

        # Move the tokens to the desired device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Get the predictions
        with torch.no_grad():
            preds = danlp_model(**inputs).logits  # type: ignore [name-defined]

        # Extract the offensive probability
        offensive_probs = torch.softmax(preds, dim=-1)[:, -1].cpu().numpy()

    # Compute the final labels
    def compute_label(offensive_prob: float):
        if offensive_prob < 0.001:
            return NOT_OFFENSIVE
        elif offensive_prob > 0.5:
            return OFFENSIVE
        else:
            return ABSTAIN

    prob_idx = 0
    for idx in range(labels.shape[0]):
        if labels[idx] == ABSTAIN:
            labels[idx] = compute_label(offensive_probs[prob_idx])
            prob_idx += 1

    return labels


@labeling_function()
def use_attack_model(record) -> np.ndarray:
    """Apply the A-ttack hatespeech detection transformer model.

    This model can be found at https://github.com/ogtal/A-ttack.

    This will mark the document as offensive if the model predicts the document as
    offensive with confidence above 50%, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Load model if it has not been loaded yet
    if "attack_tok" not in globals() or "attack_model" not in globals():
        initialise_models()

    # Set `model_max_length` if not specified
    if attack_tok.model_max_length > 100_000:  # type: ignore [name-defined]
        attack_tok.model_max_length = 512  # type: ignore [name-defined]

    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Check if any of the documents are DR answers or spam, and mark the remaining
    # documents that needs to be checked
    labels = np.maximum(labels, is_dr_answer(record))
    labels = np.maximum(labels, is_spam(record))
    labels = np.maximum(labels, is_mention(record))

    # Extract the documents
    docs = record.iloc[[idx for idx, lbl in enumerate(labels) if lbl == ABSTAIN]].text

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # Tokenise the document
        inputs = attack_tok(  # type: ignore [name-defined]
            docs.tolist(), truncation=True, padding=True, return_tensors="pt"
        )

        # Move the tokens to the desired device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        inputs.pop("token_type_ids")

        # Get the predictions
        with torch.no_grad():
            pred = attack_model(**inputs)  # type: ignore [name-defined]

        # Extract the offensive probability
        offensive_probs = torch.softmax(pred, dim=-1)[:, -1].cpu().numpy()

    # Compute the final labels
    def compute_label(offensive_prob: float):
        if offensive_prob > 0.5:
            return OFFENSIVE
        else:
            return ABSTAIN

    prob_idx = 0
    for idx in range(labels.shape[0]):
        if labels[idx] == ABSTAIN:
            labels[idx] = compute_label(offensive_probs[prob_idx])
            prob_idx += 1

    return labels


@labeling_function()
def use_tfidf_model(record) -> np.ndarray:
    """Apply the TF-IDF offensive speech detection model.

    This will mark the document as offensive if the model classifies it as offensive
    with a decision score greater than 2, and abstains otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array of the same shape as the input:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Load model if it has not been loaded yet
    if "tfidf" not in globals():
        initialise_models()

    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Check if any of the documents are DR answers or spam, and mark the remaining
    # documents that needs to be checked
    labels = np.maximum(labels, is_dr_answer(record))
    labels = np.maximum(labels, is_spam(record))
    labels = np.maximum(labels, is_mention(record))

    # Extract the documents
    docs = record.iloc[[idx for idx, lbl in enumerate(labels) if lbl == ABSTAIN]].text

    # Get the prediction score
    predicted_scores = tfidf.decision_function(docs)  # type: ignore [name-defined]

    # Compute the final labels
    def compute_label(predicted_score: float):
        if predicted_score > 2:
            return OFFENSIVE
        else:
            return ABSTAIN

    score_idx = 0
    for idx in range(len(record)):
        if labels[idx] == ABSTAIN:
            labels[idx] = compute_label(predicted_scores[score_idx])
            score_idx += 1

    return labels


@labeling_function()
def has_been_moderated(record) -> np.ndarray:
    """Check if a document has already been moderated.

    This will mark the document as offensive if it has been moderated, and abstain
    otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array of the same shape as the input:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Check if any of the documents are DR answers or spam, and mark the remaining
    # documents that needs to be checked
    labels = np.maximum(labels, is_dr_answer(record))
    labels = np.maximum(labels, is_spam(record))
    labels = np.maximum(labels, is_mention(record))

    # Extract the documents
    actions = record.iloc[
        [idx for idx, lbl in enumerate(labels) if lbl == ABSTAIN]
    ].action.tolist()

    # Compute the final labels
    def compute_label(action: str):
        if action != "none":
            return OFFENSIVE
        else:
            return ABSTAIN

    action_idx = 0
    for idx in range(len(record)):
        if labels[idx] == ABSTAIN:
            labels[idx] = compute_label(actions[action_idx])
            action_idx += 1

    return labels


@labeling_function()
def has_positive_sentiment(record) -> np.ndarray:
    """Apply a sentiment analysis model.

    This will mark the document as not offensive if the probability of the document
    being negative is less than 10%, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        NumPy array:
            The assigned labels, where 0 is not offensive, 1 is offensive, and -1 is
            abstain.
    """
    # Load model if it has not been loaded yet
    if "sent_tok" not in globals() or "sent_model" not in globals():
        initialise_models()

    # Set `model_max_length` if not specified
    if sent_tok.model_max_length > 100_000:  # type: ignore [name-defined]
        sent_tok.model_max_length = 512  # type: ignore [name-defined]

    # Initialise the array of labels
    labels = np.full(shape=len(record), fill_value=ABSTAIN, dtype=np.int8)

    # Extract the documents
    docs = record.text

    # Get the prediction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # Tokenise the document
        inputs = sent_tok(  # type: ignore [name-defined]
            docs.tolist(), truncation=True, padding=True, return_tensors="pt"
        )

        # Move the tokens to the desired device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Get the prediction
        with torch.no_grad():
            prediction = sent_model(**inputs).logits  # type: ignore [name-defined]

        # Extract the probability of the document being negative
        negative_probs = torch.softmax(prediction, dim=-1)[:, 0].cpu().numpy()

    # Compute the final labels
    def compute_label(negative_prob: float):
        if negative_prob < 0.1:
            return NOT_OFFENSIVE
        else:
            return ABSTAIN

    labels[:] = [compute_label(prob) for prob in negative_probs]

    return labels
