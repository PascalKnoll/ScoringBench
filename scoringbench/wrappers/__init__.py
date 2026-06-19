"""Probabilistic model wrappers for ScoringBench.

Re-exports all wrappers for backward compatibility. Individual wrappers
live in their own sub-modules:

    wrappers/base.py                  — DistributionPrediction, ProbabilisticWrapper
    wrappers/tabpfn.py                — TabPFNWrapper
    wrappers/tabicl.py                — TabICLWrapper
    wrappers/xgb_vector.py            — XGBVectorWrapper
"""

from .base import DistributionPrediction, ProbabilisticWrapper
from .synthefy import SynthefyWrapper

# Heavy model wrappers: import if their backing library is present, else expose
# the name as None so the package (and run_bench_regression.py's import list)
# loads without every competitor's stack installed. Only the models actually
# referenced in MODELS need their library at run time.
_OPTIONAL = [
    ("TabPFNWrapper", "tabpfn"), ("FinetuneTabPFNWrapper", "tabpfn"),
    ("TabICLWrapper", "tabicl"), ("FinetuneTabICLWrapper", "tabicl"),
    ("XGBVectorWrapper", "xgb_vector"), ("XGBQuantileVectorWrapper", "xgb_vector"),
    ("XGBLSSWrapper", "xgblss_wrapper"), ("CatBoostQuantileWrapper", "catboost_wrapper"),
    ("CrepesWrapper", "crepes_wrapper"),
    ("PytabkitRealMLPWrapper", "pytabkit"), ("PytabkitRealMLPHPOWrapper", "pytabkit"),
    ("PytabkitTabMDWrapper", "pytabkit"), ("PytabkitTabMHPOWrapper", "pytabkit"),
]
for _name, _mod in _OPTIONAL:
    try:
        _m = __import__(f"scoringbench.wrappers.{_mod}", fromlist=[_name])
        globals()[_name] = getattr(_m, _name)
    except Exception:
        globals()[_name] = None

__all__ = [
    "SynthefyWrapper",
    "DistributionPrediction",
    "ProbabilisticWrapper",
    "TabPFNWrapper",
    "FinetuneTabPFNWrapper",
    "TabICLWrapper",
    "FinetuneTabICLWrapper",
    "XGBVectorWrapper",
    "XGBQuantileVectorWrapper",
    "XGBLSSWrapper",
    "PytabkitRealMLPWrapper",
    "PytabkitRealMLPHPOWrapper",
    "PytabkitTabMDWrapper",
    "PytabkitTabMHPOWrapper",
    "CatBoostQuantileWrapper",
    "CrepesWrapper",
]
