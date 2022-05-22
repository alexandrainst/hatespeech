'''Labeling functions used for weak supervision.'''

from snorkel.labeling import labeling_function
from transformers import pipeline
import re
import joblib


# Create label names
ABSTAIN = -1
NOT_OFFENSIVE = 0
OFFENSIVE = 1


@labeling_function()
def contains_offensive_word(record) -> int:
    '''Check if the document contains an offensive word.

    This will mark the document as offensive if it contains an offensive word
    and is not an official DR answer, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document contains an offensive
            word, and -1 (abstain) otherwise.
    '''
    # Extract the document
    doc = record.text

    # Define list of offensive words
    offensive_words = [
        r'perker',
        r'snotunge',
        r'svagpisser',
        r'pik\W',
        r'fisse',
        r'retar?deret',
        r'asshole',
        r'retard',
        r'idiot',
        r'hold (dog |nu |bare )?k[æ?]ft',
        r'stodder',
        r'mongol',
        r'(m[ø?][gj]|klamme|usselt?) ?(svin|so|kost)',
        r'klaphat',
        r'kneppe',
        r'liderlig',
        r'vatpik',
        r'k[æ?]lling',
        r'fatsvag',
        r'gimpe',
        r'luder',
        r'dumb +fuck',
        r'afskum',
        r'psykopat',
        r'b[ø?]sser[ø?]v',
        r'\Wtumpe',
        r'[åa?]ndss?vag',
        r'spasser',
        r'r[ø?]vhul',
    ]

    # Mark document as offensive if it contains an offensive word, and abstain
    # otherwise
    if (not is_dr_answer(doc) and
            any(re.search(regex, doc.lower()) for regex in offensive_words)):
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def is_mention(record) -> int:
    '''Check if the document is only a mention.

    This will mark the document as not offensive if it contains a mention, and
    abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 0 (not offensive) if the document contains a mention,
            and -1 (abstain) otherwise.
    '''
    # Extract the document
    doc = record.text

    # Load the NER model
    ner = pipeline(model='saattrupdan/nbailab-base-ner-scandi', task='ner')

    # Get the PER label indices
    per_indices = [(dct['start'], dct['end']) for dct in ner(doc)
                   if dct['entity_group'] == 'PER']

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
    if doc == '':
        return NOT_OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def is_dr_answer(record) -> int:
    '''Check if the document is an official reply from DR.

    This will mark the document as not offensive if it contains an official
    reply from DR, and abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 0 (not offensive) if the document contains an
            official reply from DR, and -1 (abstain) otherwise.
    '''
    # Extract the document
    doc = record.text

    # Define phrases the DR employees tend to use
    dr_phrases = [
        r'vi har slettet din kommentar',
        r'overskrider vores retningslinjer',
        r'\W(m?vh\.?|/+) *[a-zæøå]+ *[/,]+ *dr',
    ]

    # If the document contains any of the phrases then mark it as not
    # offensive, otherwise abstain
    if any(re.search(regex, doc.lower()) for regex in dr_phrases):
        return NOT_OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def guscode_model(record) -> int:
    '''Apply the Guscode/DKbert-hatespeech-detection model.

    This will mark the document as offensive if the model classifies it as
    offensive, and will mark it as not offensive if the model classifies it as
    not offensive.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document is classified as
            offensive by the model, and 0 (not offensive) if the document is
            classified as not offensive by the model.
    '''
    # Extract the document
    doc = record.text

    # Load the model
    model = pipeline(model='Guscode/DKbert-hatespeech-detection')

    # Get the prediction
    predicted_label = model(doc)[0]['label']

    # If the predicted label is 'LABEL_0' then it is not offensive, otherwise
    # it is offensive
    if predicted_label == 'LABEL_0':
        return NOT_OFFENSIVE
    else:
        return OFFENSIVE


@labeling_function()
def danlp_electra_model(record) -> int:
    '''Apply the DaNLP/Electra-hatespeech-detection model.

    This will mark the document as offensive if the model classifies it as
    offensive, and will mark it as not offensive if the model classifies it as
    not offensive.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document is classified as
            offensive by the model, and 0 (not offensive) if the document is
            classified as not offensive by the model.
    '''
    # Extract the document
    doc = record.text

    # Load the model
    model = pipeline(model='DaNLP/da-electra-hatespeech-detection')

    # Get the prediction
    predicted_label = model(doc)[0]['label']

    # If the predicted label is 'not offensive' then it is not offensive,
    # otherwise it is offensive
    if predicted_label == 'offensive':
        return OFFENSIVE
    else:
        return NOT_OFFENSIVE


@labeling_function()
def danlp_dabert_model(record) -> int:
    '''Apply the DaNLP/da-bert-hatespeech-detection model.

    This will mark the document as offensive if the model classifies it as
    offensive, and will mark it as not offensive if the model classifies it as
    not offensive.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document is classified as
            offensive by the model, and 0 (not offensive) if the document is
            classified as not offensive by the model.
    '''
    # Extract the document
    doc = record.text

    # Load the model
    model = pipeline(model='DaNLP/da-bert-hatespeech-detection')

    # Get the prediction
    predicted_label = model(doc)[0]['label']

    # If the predicted label is 'not offensive' then it is not offensive,
    # otherwise it is offensive
    if predicted_label == 'offensive':
        return OFFENSIVE
    else:
        return NOT_OFFENSIVE


@labeling_function()
def tfidf_model(record) -> int:
    '''Apply the TF-IDF offensive speech detection model.

    This will mark the document as offensive if the model classifies it as
    offensive, and will mark it as not offensive if the model classifies it as
    not offensive.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document is classified as
            offensive by the model, and 0 (not offensive) if the document is
            classified as not offensive by the model.
    '''
    # Extract the document
    doc = record.text

    # Load the model
    pipeline = joblib.load('models/tfidf_model.bin')

    # Get the prediction
    predicted_label = pipeline.predict([doc])[0]

    # If the predicted label is 'not offensive' then it is not offensive,
    # otherwise it is offensive
    if predicted_label == 'OFF':
        return OFFENSIVE
    else:
        return NOT_OFFENSIVE


@labeling_function()
def has_been_moderated(record) -> int:
    '''Check if a document has already been moderated.

    This will mark the document as offensive if it has been moderated, and
    abstain otherwise.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document has been moderated,
            and -1 (abstain) otherwise.
    '''
    # Extract the moderation action
    action = record.action

    # If the action is not "none" then mark the document as offensive, and
    # otherwise abstain
    if action != 'none':
        return OFFENSIVE
    else:
        return ABSTAIN


@labeling_function()
def sentiment(record) -> int:
    '''Apply a sentiment analysis model.

    This will mark the document as offensive if the predicted sentiment is
    negative with a confidence of at least 0.99, and will mark it as not
    offensive if the predicted sentiment is positive with a confidence of at
    least 0.99, and otherwise abstain.

    Args:
        record:
            The record containing the document to be checked.

    Returns:
        int:
            This value is 1 (offensive) if the document has a positive
            predicted sentiment, and 0 (not offensive) if the document has a
            negative predicted sentiment, and -1 (abstain) otherwise.
    '''
    # Extract the document
    doc = record.text

    # Load the model
    model = pipeline(model='DaNLP/da-bert-tone-sentiment-polarity')

    # Get the prediction
    prediction = model(doc)[0]

    # If the predicted label is positive and the confidence is at least 0.99,
    # then mark the document as not offensive. If the predicted label is
    # negative and the confidence is at least 0.99, then mark the document as
    # offensive. Otherwise abstain.
    if prediction['label'] == 'positive' and prediction['score'] >= 0.99:
        return NOT_OFFENSIVE
    elif prediction['label'] == 'negative' and prediction['score'] >= 0.99:
        return OFFENSIVE
    else:
        return ABSTAIN
