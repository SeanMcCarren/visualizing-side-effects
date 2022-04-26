import pandas as pd
import numpy as np

dropbox_input = r"C:\Users\Sean McCarren\Documents\dropbox\input"
dropbox_items = [
    "drugs",
    "facts",
    "reactions",
]

NSIDES_input = r"D:\Data\NSIDES"
NSIDES_info = {
    "extention": "csv.xz",
    "dtype": {
        'drug_rxnorn_id': np.int64,
        'drug_1_rxnorn_id': np.int64,
        'drug_2_rxnorm_id': np.int64,
        'condition_meddra_id': np.int64,
        'A': np.int64,
        'B': np.int64,
        'C': np.int64,
        'D': np.int64,
        'PRR': np.float64,
        'PRR_error': np.float64,
        'mean_reporting_frequency': np.float64
    },
}
NSIDES_items = [
    ("OFFSIDES", {**NSIDES_info, "skiprows": [200001]}),
    ("TWOSIDES", {**NSIDES_info, "skiprows": [1001]})
]


STORE_input = r"D:\Data"
STORE_items = [
    "bio-decagon-effectcategories",
    "bio-decagon-combo",
    ("ChChSe-Decagon_polypharmacy", {"extention":"csv.gz", 'header':0, 'names':['STITCH 1', 'STITCH 2', 'Polypharmacy Side Effect', 'Side Effect Name']}),
    ("ChG-InterDecagon_targets", {"extention":"csv.gz", 'header':0, 'names':['Drug', 'Gene'], 'index_col': 'Drug'}),
    ("ChG-TargetDecagon_targets", {"extention":"csv.gz", 'header':0, 'names':['Drug', 'Gene'], 'index_col': 'Drug'}),
    ("ChSe-Decagon_monopharmacy", {"extention":"csv.gz", 'header':0, 'names':['STITCH','Individual Side Effect', 'Side Effect Name'], 'index_col': 'STITCH'}),
    ("PP-Decagon_ppi", {"extention":"csv.gz", "header": None}),
    ("Se-DoDecagon_sidefx", {"extention":"csv.gz", 'index_col': 'Side Effect'}),
]


computed_input = r"..\data"
computed_items = [
    "reaction_drug_frequency",
    ("input_data", {'index_col': 'snomed_reaction'}),
    ("sort_y", {'index_col': 'snomed_reaction'}),
    ("sort_y_temp", {'index_col': 'snomed_reaction'}),
    ("embedding", {'index_col': 'snomed_reaction'}),
    ("embedding_temp", {'index_col': 'snomed_reaction'}),
    ("emb-200", {'index_col': 'snomed_reaction'}),
    ("singles", {'index_col': 0}),
    ("doubles", {'index_col': 0}),
    ("snomed_mapping", {'index_col': 'key', 'dtype': {'found': int}}),
    ("snomed_mapping_extended", {'index_col': 'key'}),
]

data = [
    (dropbox_input, dropbox_items),
    (NSIDES_input, NSIDES_items),
    (STORE_input, STORE_items),
    (computed_input, computed_items),
]

def interp_item(item):
    kwargs = {}
    if isinstance(item, tuple) and len(item) == 2:
        name, args = item
        kwargs.update(args)
    else:
        name = item
    extention = kwargs.get('extention', 'csv')
    if 'extention' in kwargs:
        del kwargs['extention']
    return name, extention, kwargs

def load(data_name):
    for (loc, items) in data:
        for item in items:
            name, extention, kwargs = interp_item(item)
            if data_name == name:
                return pd.read_csv(f"{loc}\\{name}.{extention}", **kwargs)
    raise FileNotFoundError(name)

def save(df, data_name):
    for item in computed_items:
        name, extention, kwargs = interp_item(item)
        if name == data_name:
            df.to_csv(f"{computed_input}\\{name}.{extention}")
            return
    raise FileNotFoundError(data_name)

