from visualize_events.data import load_data
from numpy import int64

def test_load_snomed_mapping():
    mapping = load_data('snomed_mapping')

def test_snomed_mapping():
    mapping = load_data('snomed_mapping')
    assert mapping.index.dtype == int64

