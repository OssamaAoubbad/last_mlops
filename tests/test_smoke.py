from madewithml.data import clean_text
from madewithml.predict import decode, format_prob
from madewithml.serve import app


def test_clean_text_removes_noise():
    text = clean_text("This is a BERT-based NLP project!!!")

    assert text == "bert based nlp project"


def test_prediction_helpers_format_outputs():
    index_to_class = {0: "computer-vision", 1: "mlops"}

    assert decode([0, 1], index_to_class) == ["computer-vision", "mlops"]
    assert format_prob([0.25, 0.75], index_to_class) == {
        "computer-vision": 0.25,
        "mlops": 0.75,
    }


def test_fastapi_app_metadata():
    assert app.title == "Made With ML"
    assert app.version == "0.1"
