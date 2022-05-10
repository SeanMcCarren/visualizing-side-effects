from visualize_events.data import get_predictions
import pytest

def test_baseline():
    assert len(get_predictions([None])) == 1

def test_single():
    assert len(get_predictions([861])) == 1

def test_double():
    assert len(get_predictions([(36, 2772)])) == 1

def test_reverse():
    assert len(get_predictions([(2772, 36)])) == 1

def test_all():
    assert len(get_predictions([None, 861, (36, 2772)])) == 3

def test_contents():
    for df in get_predictions([None, 861, (36, 2772)]):
        assert (df.columns == ['incidence','frequency']).all()
        assert df.index.name == 'snomed_reaction'
        assert len(df) > 50
