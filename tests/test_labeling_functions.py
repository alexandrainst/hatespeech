"""Unit tests related to the labeling_functions module."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

lfs = pytest.importorskip("src.dr_hatespeech.labeling_functions")


offensive_record1 = pd.Series(
    dict(
        account="dr nyheder",
        url="https://www.facebook.com/123",
        text="Svenskere er nogle pikhoveder",
        date=dt.datetime(2020, 1, 1, 12, 0, 0),
        action="deleted",
        post_id=123,
        comment_id=np.nan,
        reply_comment_id=np.nan,
    )
)


offensive_record2 = pd.Series(
    dict(
        account="dr nyheder",
        url="https://www.facebook.com/123",
        text="kælling !!!",
        date=dt.datetime(2020, 1, 1, 12, 0, 0),
        action="deleted",
        post_id=123,
        comment_id=np.nan,
        reply_comment_id=np.nan,
    )
)

offensive_record3 = pd.Series(
    dict(
        account="dr nyheder",
        url="https://www.facebook.com/123",
        text="Ej hvor er det bare lækkert det her! Virkelig dejligt. Røvhul",
        date=dt.datetime(2020, 1, 1, 12, 0, 0),
        action="deleted",
        post_id=123,
        comment_id=np.nan,
        reply_comment_id=np.nan,
    )
)

positive_record = pd.Series(
    dict(
        account="dr nyheder",
        url="https://www.facebook.com/123",
        text="Hvor er det bare lækkert!",
        date=dt.datetime(2020, 1, 1, 12, 0, 0),
        action="none",
        post_id=123,
        comment_id=np.nan,
        reply_comment_id=np.nan,
    )
)

mention_record = pd.Series(
    dict(
        account="dr nyheder",
        url="https://www.facebook.com/123",
        text="Hans Christian",
        date=dt.datetime(2020, 1, 1, 12, 0, 0),
        action="none",
        post_id=123,
        comment_id=np.nan,
        reply_comment_id=np.nan,
    )
)

dr_answer_record = pd.Series(
    dict(
        account="dr nyheder",
        url="https://www.facebook.com/123",
        text="Hej. Vi har slettet din kommentar [fatsvage kælling] // Michael, DR",
        date=dt.datetime(2020, 1, 1, 12, 0, 0),
        action="none",
        post_id=123,
        comment_id=np.nan,
        reply_comment_id=np.nan,
    )
)

all_caps_record = pd.Series(
    dict(
        account="dr nyheder",
        url="https://www.facebook.com/123",
        text="WOW MAN HVOR ER DU KLOG",
        date=dt.datetime(2020, 1, 1, 12, 0, 0),
        action="deleted",
        post_id=123,
        comment_id=np.nan,
        reply_comment_id=np.nan,
    )
)

positive_swear_word_record = pd.Series(
    dict(
        account="dr nyheder",
        url="https://www.facebook.com/123",
        text="Fuck hvor nice man",
        date=dt.datetime(2020, 1, 1, 12, 0, 0),
        action="none",
        post_id=123,
        comment_id=np.nan,
        reply_comment_id=np.nan,
    )
)


@pytest.mark.parametrize(
    argnames="labelling_function, record, expected_result",
    argvalues=[
        (lfs.contains_offensive_word, offensive_record1, lfs.ABSTAIN),
        (lfs.contains_offensive_word, offensive_record2, lfs.OFFENSIVE),
        (lfs.contains_offensive_word, offensive_record3, lfs.OFFENSIVE),
        (lfs.contains_offensive_word, positive_record, lfs.ABSTAIN),
        (lfs.contains_offensive_word, mention_record, lfs.ABSTAIN),
        (lfs.contains_offensive_word, dr_answer_record, lfs.NOT_OFFENSIVE),
        (lfs.contains_offensive_word, all_caps_record, lfs.ABSTAIN),
        (lfs.contains_offensive_word, positive_swear_word_record, lfs.ABSTAIN),
        (lfs.is_all_caps, offensive_record1, lfs.ABSTAIN),
        (lfs.is_all_caps, offensive_record2, lfs.ABSTAIN),
        (lfs.is_all_caps, offensive_record3, lfs.ABSTAIN),
        (lfs.is_all_caps, positive_record, lfs.ABSTAIN),
        (lfs.is_all_caps, mention_record, lfs.ABSTAIN),
        (lfs.is_all_caps, dr_answer_record, lfs.ABSTAIN),
        (lfs.is_all_caps, all_caps_record, lfs.OFFENSIVE),
        (lfs.is_all_caps, positive_swear_word_record, lfs.ABSTAIN),
        (lfs.contains_positive_swear_word, offensive_record1, lfs.ABSTAIN),
        (lfs.contains_positive_swear_word, offensive_record2, lfs.ABSTAIN),
        (lfs.contains_positive_swear_word, offensive_record3, lfs.ABSTAIN),
        (lfs.contains_positive_swear_word, positive_record, lfs.ABSTAIN),
        (lfs.contains_positive_swear_word, mention_record, lfs.ABSTAIN),
        (lfs.contains_positive_swear_word, dr_answer_record, lfs.ABSTAIN),
        (lfs.contains_positive_swear_word, all_caps_record, lfs.ABSTAIN),
        (
            lfs.contains_positive_swear_word,
            positive_swear_word_record,
            lfs.NOT_OFFENSIVE,
        ),
        (lfs.is_mention, offensive_record1, lfs.ABSTAIN),
        (lfs.is_mention, offensive_record2, lfs.ABSTAIN),
        (lfs.is_mention, offensive_record3, lfs.ABSTAIN),
        (lfs.is_mention, positive_record, lfs.ABSTAIN),
        (lfs.is_mention, mention_record, lfs.NOT_OFFENSIVE),
        (lfs.is_mention, dr_answer_record, lfs.ABSTAIN),
        (lfs.is_mention, all_caps_record, lfs.ABSTAIN),
        (lfs.is_mention, positive_swear_word_record, lfs.ABSTAIN),
        (lfs.is_dr_answer, offensive_record1, lfs.ABSTAIN),
        (lfs.is_dr_answer, offensive_record2, lfs.ABSTAIN),
        (lfs.is_dr_answer, offensive_record3, lfs.ABSTAIN),
        (lfs.is_dr_answer, positive_record, lfs.ABSTAIN),
        (lfs.is_dr_answer, mention_record, lfs.ABSTAIN),
        (lfs.is_dr_answer, dr_answer_record, lfs.NOT_OFFENSIVE),
        (lfs.is_dr_answer, all_caps_record, lfs.ABSTAIN),
        (lfs.is_dr_answer, positive_swear_word_record, lfs.ABSTAIN),
        (lfs.use_hatespeech_model, offensive_record1, lfs.OFFENSIVE),
        (lfs.use_hatespeech_model, offensive_record2, lfs.ABSTAIN),
        (lfs.use_hatespeech_model, offensive_record3, lfs.NOT_OFFENSIVE),
        (lfs.use_hatespeech_model, positive_record, lfs.NOT_OFFENSIVE),
        (lfs.use_hatespeech_model, mention_record, lfs.NOT_OFFENSIVE),
        (lfs.use_hatespeech_model, dr_answer_record, lfs.NOT_OFFENSIVE),
        (lfs.use_hatespeech_model, all_caps_record, lfs.ABSTAIN),
        (lfs.use_hatespeech_model, positive_swear_word_record, lfs.ABSTAIN),
        (lfs.use_tfidf_model, offensive_record1, lfs.ABSTAIN),
        (lfs.use_tfidf_model, offensive_record2, lfs.OFFENSIVE),
        (lfs.use_tfidf_model, offensive_record3, lfs.ABSTAIN),
        (lfs.use_tfidf_model, positive_record, lfs.ABSTAIN),
        (lfs.use_tfidf_model, mention_record, lfs.ABSTAIN),
        (lfs.use_tfidf_model, dr_answer_record, lfs.NOT_OFFENSIVE),
        (lfs.use_tfidf_model, all_caps_record, lfs.ABSTAIN),
        (lfs.use_tfidf_model, positive_swear_word_record, lfs.OFFENSIVE),
        (lfs.has_been_moderated, offensive_record1, lfs.OFFENSIVE),
        (lfs.has_been_moderated, offensive_record2, lfs.OFFENSIVE),
        (lfs.has_been_moderated, offensive_record3, lfs.OFFENSIVE),
        (lfs.has_been_moderated, positive_record, lfs.ABSTAIN),
        (lfs.has_been_moderated, mention_record, lfs.ABSTAIN),
        (lfs.has_been_moderated, dr_answer_record, lfs.NOT_OFFENSIVE),
        (lfs.has_been_moderated, all_caps_record, lfs.OFFENSIVE),
        (lfs.has_been_moderated, positive_swear_word_record, lfs.ABSTAIN),
        (lfs.has_positive_sentiment, offensive_record1, lfs.ABSTAIN),
        (lfs.has_positive_sentiment, offensive_record2, lfs.ABSTAIN),
        (lfs.has_positive_sentiment, offensive_record3, lfs.NOT_OFFENSIVE),
        (lfs.has_positive_sentiment, positive_record, lfs.NOT_OFFENSIVE),
        (lfs.has_positive_sentiment, mention_record, lfs.ABSTAIN),
        (lfs.has_positive_sentiment, dr_answer_record, lfs.ABSTAIN),
        (lfs.has_positive_sentiment, all_caps_record, lfs.ABSTAIN),
        (lfs.has_positive_sentiment, positive_swear_word_record, lfs.ABSTAIN),
    ],
    ids=[
        "contains_offensive_word_offensive_record1",
        "contains_offensive_word_offensive_record2",
        "contains_offensive_word_offensive_record3",
        "contains_offensive_word_positive_record",
        "contains_offensive_word_mention",
        "contains_offensive_word_dr_answer",
        "contains_offensive_word_all_caps",
        "contains_offensive_word_positive_swear_word",
        "is_all_caps_offensive_record1",
        "is_all_caps_offensive_record2",
        "is_all_caps_offensive_record3",
        "is_all_caps_positive_record",
        "is_all_caps_mention",
        "is_all_caps_dr_answer",
        "is_all_caps_all_caps",
        "is_all_caps_positive_swear_word",
        "contains_positive_swear_word_offensive_record1",
        "contains_positive_swear_word_offensive_record2",
        "contains_positive_swear_word_offensive_record3",
        "contains_positive_swear_word_positive_record",
        "contains_positive_swear_word_mention",
        "contains_positive_swear_word_dr_answer",
        "contains_positive_swear_word_all_caps",
        "contains_positive_swear_word_positive_swear_word",
        "is_mention_offensive_record1",
        "is_mention_offensive_record2",
        "is_mention_offensive_record3",
        "is_mention_positive_record",
        "is_mention_mention",
        "is_mention_dr_answer",
        "is_mention_all_caps",
        "is_mention_positive_swear_word",
        "is_dr_answer_offensive_record1",
        "is_dr_answer_offensive_record2",
        "is_dr_answer_offensive_record3",
        "is_dr_answer_positive_record",
        "is_dr_answer_mention",
        "is_dr_answer_dr_answer",
        "is_dr_answer_all_caps",
        "is_dr_answer_positive_swear_word",
        "use_hatespeech_model_offensive_record1",
        "use_hatespeech_model_offensive_record2",
        "use_hatespeech_model_offensive_record3",
        "use_hatespeech_model_positive_record",
        "use_hatespeech_model_mention",
        "use_hatespeech_model_dr_answer",
        "use_hatespeech_model_all_caps",
        "use_hatespeech_model_positive_swear_word",
        "use_tfidf_model_offensive_record1",
        "use_tfidf_model_offensive_record2",
        "use_tfidf_model_offensive_record3",
        "use_tfidf_model_positive_record",
        "use_tfidf_model_mention",
        "use_tfidf_model_dr_answer",
        "use_tfidf_model_all_caps",
        "use_tfidf_model_positive_swear_word",
        "has_been_moderated_offensive_record1",
        "has_been_moderated_offensive_record2",
        "has_been_moderated_offensive_record3",
        "has_been_moderated_positive_record",
        "has_been_moderated_mention",
        "has_been_moderated_dr_answer",
        "has_been_moderated_all_caps",
        "has_been_moderated_positive_swear_word",
        "has_positive_sentiment_offensive_record1",
        "has_positive_sentiment_offensive_record2",
        "has_positive_sentiment_offensive_record3",
        "has_positive_sentiment_positive_record",
        "has_positive_sentiment_mention",
        "has_positive_sentiment_dr_answer",
        "has_positive_sentiment_all_caps",
        "has_positive_sentiment_positive_swear_word",
    ],
)
def test_labelling_function(labelling_function, record, expected_result):
    assert labelling_function(record) == expected_result
