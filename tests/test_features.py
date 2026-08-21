from src.features import FEATURES, LEAKAGE_COLUMNS

def test_no_post_outcome_leakage():
    assert not set(FEATURES).intersection(LEAKAGE_COLUMNS)

def test_features_are_unique():
    assert len(FEATURES) == len(set(FEATURES))
