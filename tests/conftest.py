import pytest

SAMPLE_ANSWER_VOCAB = [
    "yes",
    "no",
    "bone",
    "bone, calvarium",
    "gastrointestinal",
    "gastrointestinal system",
    "lung",
    "red",
    "blue",
    "large",
    "small",
]

SAMPLE_ANSWER_TO_IDX = {a: i for i, a in enumerate(SAMPLE_ANSWER_VOCAB)}


@pytest.fixture
def answer_vocab():
    return list(SAMPLE_ANSWER_VOCAB)


@pytest.fixture
def answer_to_idx():
    return dict(SAMPLE_ANSWER_TO_IDX)
