"""Smoke tests for the training -> serving handoff.

A pickle is only loadable in an environment compatible with the one that built it.
This script proves, on every push, that the artifacts and the pinned requirements
still agree -- and that every model produces a prediction of the expected shape.

Run:  python tests/test_artifacts.py
"""

import os
import sys
import py_compile

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")

EXPECTED = [
    "visitmode_classifier.pkl",
    "rating_regressor.pkl",
    "satisfaction_classifier.pkl",
    "aggregates.pkl",
    "lookups.pkl",
    "recommender.pkl",
    "attr_country.pkl",
    "tourism_clean.csv",
    "classification_results.csv",
    "regression_results.csv",
    "recommender_results.csv",
]

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


print("1. artifact files present")
for fname in EXPECTED:
    check(fname, os.path.exists(os.path.join(ART, fname)))

if failures:
    print("\nMissing artifacts; cannot continue.")
    sys.exit(1)

print("\n2. artifacts deserialise")
clf = joblib.load(os.path.join(ART, "visitmode_classifier.pkl"))
reg = joblib.load(os.path.join(ART, "rating_regressor.pkl"))
sat = joblib.load(os.path.join(ART, "satisfaction_classifier.pkl"))
agg = joblib.load(os.path.join(ART, "aggregates.pkl"))
lk = joblib.load(os.path.join(ART, "lookups.pkl"))
rec = joblib.load(os.path.join(ART, "recommender.pkl"))
ac = joblib.load(os.path.join(ART, "attr_country.pkl"))
check("all seven pickles loaded", True)

print("\n3. config schema")
for key in ("classes", "mode_names", "clf_feats", "reg_feats", "sat_threshold", "n_folds"):
    check(f"lookups['{key}']", key in lk)
check("5 visit-mode classes", len(lk["classes"]) == 5, str(lk["classes"]))
check("smoothing constants stored",
      "_global_mean" in agg and "_mode_smooth" in agg and "_smooth_r" in agg)
check("global mean is a plausible rating",
      1.0 <= agg["_global_mean"] <= 5.0, f"{agg['_global_mean']:.4f}")

print("\n4. models predict")
Xc = pd.DataFrame([{f: 0.0 for f in lk["clf_feats"]}])[lk["clf_feats"]]
Xr = pd.DataFrame([{f: 0.0 for f in lk["reg_feats"]}])[lk["reg_feats"]]

proba = clf.predict_proba(Xc)
check("classifier output shape", proba.shape == (1, len(lk["classes"])), str(proba.shape))
check("classifier probabilities sum to 1", abs(proba.sum() - 1.0) < 1e-6)

yhat = float(np.clip(reg.predict(Xr)[0], 1, 5))
check("regressor returns a rating in [1, 5]", 1.0 <= yhat <= 5.0, f"{yhat:.3f}")

risk = float(sat.predict_proba(Xr)[0, 0])
check("satisfaction risk in [0, 1]", 0.0 <= risk <= 1.0, f"{risk:.3f}")

print("\n5. recommender payload")
items = rec["items"]
check("30 attractions in the catalogue", len(items) == 30, str(len(items)))
check("continent table columns match items", list(rec["cont_logp"].columns) == items)
check("global fallback row is the right length", len(rec["global_logp"]) == len(items))
check("attraction metadata covers every item", len(rec["meta"]) == len(items))
check("attr_country covers every item", all(int(i) in ac for i in items))

print("\n6. cleaned dataset")
df = pd.read_csv(os.path.join(ART, "tourism_clean.csv"))
check("no null values", int(df.isna().sum().sum()) == 0)
check("ratings within 1-5", bool(df["Rating"].between(1, 5).all()))
check("visit modes within 1-5", bool(df["VisitMode"].between(1, 5).all()))
check("VisitModeName populated", bool(df["VisitModeName"].notna().all()))
check("row count", len(df) > 50000, f"{len(df):,} rows")

print("\n7. app.py compiles")
try:
    py_compile.compile(os.path.join(ROOT, "app.py"), doraise=True)
    check("app.py syntax", True)
except py_compile.PyCompileError as exc:
    check("app.py syntax", False, str(exc))

print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)} check(s) -> {', '.join(failures)}")
    sys.exit(1)
print("All checks passed. Artifacts are consistent with the pinned environment.")
