from symbolic.query_parser import parse_question


def test_attribute_question_color():
    q = parse_question("what color is the lesion?", ["lesion"])
    assert q.qtype == "attribute"
    assert q.attribute == "color"


def test_attribute_question_shape():
    q = parse_question("what shape are the cells?", ["cells"])
    assert q.qtype == "attribute"
    assert q.attribute == "shape"


def test_count_question():
    q = parse_question("how many nuclei are present?", ["nuclei"])
    assert q.qtype == "count"


def test_yes_no_question():
    q = parse_question("is this a malignant tumor?", ["malignant", "tumor"])
    assert q.qtype == "yes_no"


def test_identity_question():
    q = parse_question("what is shown in this image?")
    assert q.qtype == "identity"


def test_location_question():
    q = parse_question("where is the abnormality located?")
    assert q.qtype == "location"


def test_which_organ_is_location():
    q = parse_question("which organ is affected by the lesion?")
    assert q.qtype == "location"


def test_attribute_beats_identity_priority():
    q = parse_question("what color is the lesion?", ["lesion"])
    assert q.qtype == "attribute"


def test_target_extraction_longest_word():
    q = parse_question("what is the lesion?", ["lesion", "malignant"])
    assert q.target == "lesion"


def test_target_extraction_without_vocab():
    q = parse_question("what is shown in this image?")
    assert q.target == ""


def test_unmatched_question_defaults_to_identity():
    q = parse_question("describe the specimen")
    assert q.qtype == "identity"
