from pandas import DataFrame, read_csv
from .data_loader import load_data
from pathlib import Path
import os
from typing import List, Tuple, Union
from numpy import all, isin, int64, sqrt, divide, multiply
from pandas import DataFrame
from statsmodels.stats.proportion import proportion_confint

DrugPair = Union[Tuple[int, int], int, None]
Prediction = DataFrame

_facts = None

# TODO extract singles and doubles from facts to reduce dependency on data
_singles = None
_doubles = None
_drugs = None

DATA_DIR = Path(__file__).parent / 'simple_predictions_cache'

def _exists(path):
    return os.path.isfile(path)

def _get_path(drug_pair: DrugPair):
    if drug_pair is None:
        stringified = 'none'
    elif isinstance(drug_pair, tuple):
        if drug_pair[1] < drug_pair[0]:
            drug_pair = (drug_pair[1], drug_pair[0])
        stringified = "_".join([str(d) for d in drug_pair])
    else:
        stringified = str(drug_pair)
    return DATA_DIR / stringified

def _load(path: Path):
    return read_csv(path, index_col='snomed_reaction', dtype={'snomed_reaction': int64, 'frequency': float, 'error': float})

def _save(data, path: Path):
    data.to_csv(path)

def frequency_prediction_from_data(reactions):
    # Assuming a DF with cases and reactions
    reactions = reactions[['case_id', 'snomed_reaction']].drop_duplicates()
    N_cases = reactions['case_id'].nunique()
    incidence = reactions.groupby('snomed_reaction')['case_id'].count()
    frequency = incidence / N_cases

    ci_low, ci_upp = proportion_confint(incidence, N_cases, alpha=0.05, method='jeffreys')

    ci_low[ci_low < 1 / N_cases] = 0
    # Normal approximation of CI: bad!
    # z = 1.96
    # p_hat = frequency.values
    # numerator = multiply(p_hat, (1-p_hat))
    # # lower_CI = p_hat - z * sqrt(divide(numerator,N_cases))
    # error = z * sqrt(divide(numerator,N_cases))

    return DataFrame({'frequency': frequency, 'ci_low': ci_low, 'ci_upp': ci_upp, 'incidence': incidence})

def _compute_prediction_two_drugs(drug_pair):
    global _doubles, _drugs
    if _doubles is None:
        _doubles = load_data('doubles')
        drug_1 = _doubles.groupby(['case_id'])['tox_drug_id'].min()
        drug_2 = _doubles.groupby(['case_id'])['tox_drug_id'].max()
        assert (drug_1 != drug_2).all()
        assert (drug_1.index == drug_2.index).all()
        _drugs = DataFrame({'drug_1': drug_1, 'drug_2': drug_2})

    if drug_pair[1] < drug_pair[0]:
        drug_pair = (drug_pair[1], drug_pair[0])

    cases_with_drugpair = _drugs[all(_drugs.values == drug_pair, axis=1)].index
    reaction_with_drugpair = _doubles[isin(_doubles['case_id'],cases_with_drugpair)]
    return frequency_prediction_from_data(reaction_with_drugpair)
    

def _compute_prediction_single_drug(drug):
    global _singles
    if _singles is None:
        _singles = load_data('singles')
    reactions_with_drug = _singles[_singles['tox_drug_id'] == drug]
    return frequency_prediction_from_data(reactions_with_drug)

def _compute_predictions_baseline():
    global _facts
    if _facts is None:
        _facts = load_data('facts')
    return frequency_prediction_from_data(_facts)

def get_predictions(drug_pairs: Union[DrugPair,List[DrugPair]], predictor='frequency') -> List[DataFrame]:
    """
    Compute predictions from data.
    """
    if predictor not in ['frequency', 'ci_low', 'ci_upp', 'incidence']:
        raise ValueError("predictor method should be one of frequency, ci_low, or ci_upp")
    if drug_pairs is None or isinstance(drug_pairs, (tuple, int)):
        path = _get_path(drug_pairs)
        if _exists(path):
            data = _load(path)
        else:
            if drug_pairs is None:
                data = _compute_predictions_baseline()
            elif isinstance(drug_pairs, tuple):
                data = _compute_prediction_two_drugs(drug_pairs)
            else:
                data = _compute_prediction_single_drug(drug_pairs)
            _save(data, path)
        return data[predictor]
    elif isinstance(drug_pairs, list):
        return [get_predictions(drug_pair, predictor=predictor) for drug_pair in drug_pairs]
    else:
        raise ValueError("Parameter drug_pairs is not a DrugPair or a list of DrugPair")
