from visualize_events.data import get_predictions
import pytest

def test_baseline():
    assert get_predictions(None) is not None

def test_single():
    assert get_predictions(861) is not None

def test_double():
    assert get_predictions((36, 2772)) is not None

def test_reverse():
    assert get_predictions((2772, 36)) is get_predictions((36, 2772))

def test_all():
    assert len(get_predictions([None, 861, (36, 2772)])) == 3

def test_contents():
    for df in get_predictions([None, 861, (36, 2772)]):
        assert (df.columns == ['incidence','frequency']).all()
        assert df.index.name == 'snomed_reaction'
        assert len(df) > 50
