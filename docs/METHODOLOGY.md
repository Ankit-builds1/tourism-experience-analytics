# Methodology

A technical account of how the models were built, what broke along the way, and how each ceiling was measured. Written to be reproducible and to be argued with.

---

## 1. Data preparation

### 1.1 Source tables

Nine Excel tables in a star schema around `Transaction`:

| Table | Rows | Key | Notes |
|---|---|---|---|
| `Transaction` | 52,930 | `TransactionId` | fact table: user, year, month, mode, attraction, rating |
| `User` | 33,530 | `UserId` | continent / region / country / city ids |
| `City` | 9,143 | `CityId` | |
| `Updated_Item` | 1,698 | `AttractionId` | supersedes `Item` (999 rows) |
| `Country` | 165 | `CountryId` | |
| `Region` | 21 | `RegionId` | padded to 1,000 rows in the file |
| `Type` | 17 | `AttractionTypeId` | |
| `Mode` | 6 | `VisitModeId` | |
| `Continent` | 6 | `ContinentId` | padded to 1,000 rows in the file |

### 1.2 Defects found and handled

| Defect | Detection | Fix |
|---|---|---|
| `Continent.xlsx` / `Region.xlsx` padded with ~980 blank rows | `df.shape` vs non-null count | `dropna(how="all")` on load |
| `Mode (1).xlsx` byte-identical duplicate of `Mode.xlsx` | file listing | filename-length tiebreak in the loader |
| `Item.xlsx` (999) superseded by `Updated_Item.xlsx` (1,698) | row count + key overlap | prefer `Updated_Item`, fall back to `Item` |
| A numeric-coercion pass destroyed `Mode.VisitMode` (text names → `NaN`) | downstream `value_counts()` returned an empty Series | rename the text column **before** the coercion pass |
| Duplicate `(user, attraction)` transactions — user 60799 rated Merapi Volcano **37 times** | `groupby` size vs nunique | `aggfunc="mean"` when building the user-item matrix |
| `Country.Country` inconsistently cased (`NIGERIA`) | value inspection | `.str.strip().str.title()` |
| `VisitMode == 0` means "unknown" per the `Mode` table | join against `Mode` | filtered out |

Row count after cleaning: **52,930 → 52,922** (8 rows dropped by the rating / visit-mode / calendar range filters; every other defect was structural rather than row-level). Every merge asserts `len(df) == len(transaction)` to catch a fan-out before the row filters run.

### 1.3 Engineered base features

`VisitQuarter`, `Season`, cyclic `MonthSin` / `MonthCos`, and `IsDomestic` (visitor's country equals the attraction's country).

---

## 2. The leakage investigation

This is the part worth reading in full. It cost several iterations and is the difference between the model working and being anti-predictive.

### 2.1 The symptom

With leave-one-out (LOO) target encoding, the models that should have been strongest were the worst:

| Model | R² | Reading |
|---|---|---|
| Ridge | **+0.126** | fine |
| Random Forest | +0.031 | poor |
| HistGradientBoosting | −0.020 | worse than predicting the mean |
| XGBoost | **−0.642** | catastrophically wrong |

Simultaneously, the satisfaction classifier reached ROC-AUC **0.482** — *below* chance — while predicting a single class for all 10,585 test rows.

### 2.2 The first (wrong) diagnosis

Initial read: overfitting. The trees were deep (`max_depth=8`), numerous (800 rounds), and the feature set contained raw high-cardinality IDs (`CityId` with 5,546 levels).

Regularising helped Random Forest (−0.025 → +0.031) but left HistGB at −0.020 and XGBoost at −0.642. **A depth-4, lr-0.03, `min_child_weight=50`, `reg_lambda=5` XGBoost cannot reach R² = −0.64 by overfitting.** Overfitting was real but not the root cause.

### 2.3 The actual cause

LOO encoding computes, for a training row:

```
attr_rt = (S - r) / (n - 1)
```

where `S` is the attraction's rating sum and `n` its row count. Within a single attraction group **both are constants**, so `attr_rt` is an exactly decreasing linear function of that row's own target `r`. A model that splits finely enough on `attr_rt` reads the target off backwards.

The size of the back-channel is `4 / (n - 1)` — the spread between `r = 1` and `r = 5`:

| Attraction | `n` in train | LOO spread | Comparable to the 1.3 spread *across* attractions? |
|---|---|---|---|
| 640 | ~10,500 | 0.0004 | no |
| 748 | ~4,650 | 0.0009 | no |
| 928 | ~22 | **0.19** | **yes** |
| small countries / cities | < 30 | **0.13 – 0.4** | **yes** |

So for small groups the encoding varied *more* from the leak than from genuine between-group differences. At training time the model learns "higher `attr_rt` → lower rating". At test time the relationship inverts (test rows were never in `S`), so those splits predict backwards.

This explains the exact ordering of the damage:

| Model | Damage | Why |
|---|---|---|
| XGBoost | −0.642 | most aggressive at finding fine splits |
| HistGradientBoosting | −0.020 | 255-bin quantisation blunts the fine structure |
| Random Forest | +0.031 | bagging + feature subsampling dilutes it |
| Ridge / Linear | unaffected | a tiny linear perturbation absorbed into one coefficient |

### 2.4 The fix — out-of-fold encoding

Each training row's encoding is computed from a **disjoint** set of rows, so no deterministic self-relationship can exist:

```python
FOLD = np.zeros(len(train), dtype=int)
for f, (_, idx) in enumerate(KFold(5, shuffle=True, random_state=42).split(train)):
    FOLD[idx] = f

for f in range(N_FOLDS):
    hold = FOLD == f
    g = tr.loc[~hold].groupby(key)["Rating"].agg(["sum", "count"])   # other folds only
    s = tr.loc[hold, key].map(g["sum"]).fillna(0)
    c = tr.loc[hold, key].map(g["count"]).fillna(0)
    enc[hold] = (s + m * GLOBAL_MEAN) / (c + m)                      # smoothed
```

Test rows use the full-train aggregate. Counts on the test side are scaled by `(k-1)/k` so the `*_n` features match the distribution the model was trained on — an easy detail to miss that quietly shifts every count-based split threshold by 25%.

### 2.5 Verification

An automated check now runs after feature construction. A leaking encoding shows an *inflated* train correlation and a *collapsed or inverted* test correlation:

```
Leak check - rating encodings (signs must match, magnitudes should be close):
  user_rt      train=+0.1656  test=+0.1741
  attr_rt      train=+0.3028  test=+0.2884
  country_rt   train=+0.0852  test=+0.0961
  city_rt      train=+0.0875  test=+0.0971
  type_rt      train=+0.2624  test=+0.2543
  cont_rt      train=+0.0410  test=+0.0322
```

All six agree in sign and to within 0.015 in magnitude. Results after the fix:

| Model | LOO | Out-of-fold |
|---|---|---|
| XGBoost | −0.642 | **+0.145** |
| HistGradientBoosting | −0.020 | **+0.141** |
| Random Forest | +0.031 | **+0.130** |
| Ridge | +0.126 | +0.118 |
| Satisfaction ROC-AUC | 0.482 | **0.712** |

Ridge losing 0.008 is the tell that it had been taking a small amount of free lift from the leak too.

---

## 3. Smoothing

Additive (Laplace) smoothing:

```
encoding = (group_sum + m * global_mean) / (group_count + m)
```

`m` is the number of imaginary observations drawn from the global distribution mixed into every group. It has to be chosen per key, because *how much a single observation tells you* differs per key.

### 3.1 Why a uniform `m` was wrong

EDA measured that repeat users keep the same visit mode **85%** of the time. With a uniform `m = 5`, a user with one prior "Couples" visit gets:

```
p(Couples) = (1 + 5 * 0.41) / (1 + 5) = 0.508
```

— barely above the 0.41 global rate. The strongest feature in the dataset was being outvoted 5-to-1 by the prior. With `m = 1`:

```
p(Couples) = (1 + 0.41) / (1 + 1) = 0.705
```

### 3.2 Final values

| Key | `m` | Rationale |
|---|---|---|
| `umode` (user) | 1.0 | 85% mode-consistent; one visit is strong evidence |
| `citymode` | 3.0 | 5,546 cities, moderate group sizes |
| `amode`, `cmode` | 10.0 | thousands of rows per group, already stable |
| `contmode`, `tmode` | 20.0 | 5 continents / 17 types — very large groups |
| all rating encodings | 20.0 | a single rating is noisy regardless of key |

Measured effect: the mean peak of the user visit-mode prior rose from **0.51 → 0.664**. Downstream accuracy gain was **+0.16 points** (62.74% → 62.90%) — real but smaller than the sharpening suggests, because only 51% of test rows have any user history at all.

### 3.3 Prior quality by key

Agreement of each prior's argmax with the true visit mode, restricted to rows that have history:

| Key | `m` | Train | Test | Coverage |
|---|---|---|---|---|
| `umode` | 1.0 | 0.739 | **0.750** | 51% |
| `amode` | 10.0 | 0.469 | 0.462 | 100% |
| `tmode` | 20.0 | 0.468 | 0.462 | 100% |
| `citymode` | 3.0 | 0.441 | 0.445 | 95% |
| `cmode` | 10.0 | 0.425 | 0.431 | 100% |
| `contmode` | 20.0 | 0.413 | 0.417 | 100% |
| *majority baseline* | — | — | *0.408* | — |

**One feature carries the model.** Geography is worth 1–6 points over guessing "Couples" for everyone; user history is worth 34.

---

## 4. Measuring the ceilings

Both ceilings were measured *before* concluding the models were done, which is what makes "we stopped here" a finding rather than an admission.

### 4.1 Regression: R² ≈ 0.26

Decompose rating variance into within-user and total:

```
global variance      = 0.9420
within-user variance = 0.6964   (multi-visit users only)
R² ceiling from a perfect user model = 1 - 0.6964/0.9420 = 0.261
```

A model that knew each user's true mean exactly could explain at most 26% of rating variance. Measured R² of 0.1448 is **56% of that**. Reporting "R² = 0.14" without this context invites the wrong conclusion.

### 4.2 Classification: accuracy ≈ 0.628

From the prior-quality table above:

| Row group | Share of test | Best achievable | Contribution |
|---|---|---|---|
| Has user history | 51% | ~0.78 (prior alone gives 0.750) | 0.398 |
| No user history | 49% | ~0.47 (best demographic cell = 51.4%) | 0.230 |
| | | **ceiling** | **≈ 0.628** |

Measured best: **0.6290**. Six model families spanning logistic regression, bagging and three boosting implementations landed within **0.91 points** of each other. When independent model classes converge *and* the number matches a coverage × accuracy decomposition, the limit is informational.

---

## 5. Models

### 5.1 Feature sets

Raw high-cardinality IDs (`AttractionId`, `AttractionCityId`, `CityId`, `CountryId`, `RegionId`) are **excluded** from every model. They are already represented by their target encodings, so including them adds no information and gives trees a memorisation surface. `AttractionTypeId` (17 levels) and `ContinentId` (6) are retained.

- **Regression** — 22 features: calendar + cyclic month, `VisitMode`, `IsDomestic`, 6 rating encodings with counts, 3 deviation features (`user_vs_global`, `attr_vs_global`, `user_attr_gap`)
- **Classification** — 54 features: the above minus `VisitMode`, plus `Rating`, plus 6 × (5 class probabilities + count + argmax) visit-mode priors

### 5.2 Hyperparameters

Deliberately conservative. With a true R² ceiling of 0.26 there is very little signal, and aggressive trees find noise:

```python
XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4,
             min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=5.0, tree_method="hist")

LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
               min_child_samples=100, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, reg_lambda=5.0)
```

### 5.3 Satisfaction model and threshold

`class_weight="balanced"` is required — without it the model predicts "Satisfied" for every row and reports the 78.87% base rate as its accuracy. With balancing, the 0.5 cutoff is no longer accuracy-optimal, so the threshold becomes an explicit choice:

| Threshold | Accuracy | Recall (dissatisfied) | Precision (dissatisfied) | Share flagged |
|---|---|---|---|---|
| 0.25 | **79.60%** | 0.148 | 0.567 | 5.5% |
| 0.35 | 76.85% | 0.331 | 0.437 | 16.0% |
| **0.50** | 66.78% | **0.613** | 0.341 | 38.0% |
| 0.65 | 44.27% | 0.904 | 0.262 | 72.9% |

ROC-AUC (0.7123) is threshold-free and is the model-quality metric. The threshold is a business decision: for service recovery, a missed complaint costs more than a wasted follow-up, so **0.50** is recommended over the accuracy-optimal 0.25.

---

## 6. Recommendation — what was rejected and why

### 6.1 Models built

Item-item collaborative filtering (mean-centred cosine), truncated SVD (12 factors), content-based (TF-IDF over attraction name / type / city / address, plus one-hot categoricals), a tuned CF/content hybrid, global popularity × quality, and popularity conditioned on continent and on country.

### 6.2 Results

| Recommender | HitRate@5 | MAP@5 |
|---|---|---|
| **Popularity by continent** | **0.4978** | **0.3099** |
| Blend (pop 2.0 / geo 0.5 / CF 0.0) | 0.4969 | 0.2896 |
| Popularity (global) | 0.4936 | 0.2434 |
| Popularity by country | 0.4911 | 0.3037 |
| Content-based | 0.3025 | 0.1035 |
| Item-item CF | 0.2902 | 0.1015 |
| SVD | 0.2893 | 0.1009 |
| CF/content hybrid | 0.2891 | 0.1008 |
| CF (damped user mean) | 0.1671 | 0.0593 |
| *random* | *0.1667* | — |

On RMSE, an **item-mean baseline (0.9279) beat every recommender** (0.9988 – 1.0169).

### 6.3 Why CF fails here — mechanically

The matrix is 28,708 × 30 at **4.30% density**; users average **1.3 ratings**.

For a user with exactly one rating, mean-centring gives `C[u] = rating − user_mean = 0` in every column. The numerator `C @ S` is identically zero and the prediction collapses to that single rating for all 30 items. The model cannot differentiate items for the majority of users. Damping the user mean fixes the algebra but leaves only noise, which is why damped CF scored *exactly* at chance.

Two independent confirmations:

1. The hybrid weight search chose **α = 0.7 toward popularity** on rating RMSE.
2. The ranking weight search chose **w = 5.0**, the largest value in its grid — it wanted to drown the CF signal as much as it was allowed to. A tuner running to the edge of its range is telling you the down-weighted component contributes nothing.

### 6.4 Why continent beats country

153 countries fragment 42,337 training rows into groups too small to estimate; 5 continents keep enough data per group. Country scores *below* global popularity (0.4911 vs 0.4936) despite `m = 50` smoothing.

### 6.5 Why the gain shows in MAP, not HitRate

With 30 items and 5 slots, the recommender already shows 17% of the catalogue, so *which* items appear barely changes. Every continent surfaces the same six Bali landmarks:

```
Global:         Waterbom, Monkey Forest, Tegalalang, Uluwatu, Tanah Lot
Australia:      Waterbom, Monkey Forest, Tegalalang
United Kingdom: Monkey Forest, Tegalalang, Waterbom
India:          Tanah Lot, Monkey Forest, Uluwatu      <- the one real deviation
```

HitRate is binary and saturates. MAP rewards *position* — moving a hit from rank 4 to rank 1 raises average precision from 0.25 to 1.00 without changing hit rate at all. Hence +0.4 points of HitRate but **+27% MAP**.

---

## 7. Evaluation discipline

- **Stratified 80/20 split** on `VisitMode`, fixed `random_state=42`
- **Every hyperparameter tuned on a train-only validation split.** The hybrid weight α and the popularity weight w were selected on a 15% slice of `train`, never on test. With 6 candidate rankers × 8 candidate weights, tuning on test would have made the winner partly noise
- **Every metric reported against its baseline.** Majority class for classification, train-mean for regression, random for ranking, base rate for the binary task
- **No metric quoted that merely reproduces the base rate.** "Accuracy within ±1 star" (78.7%) and "satisfaction accuracy" (78.9%) were both computed, found to match their baselines (78.9% and 78.87%), and dropped from the reported results

---

## 8. Training → serving boundary

`artifacts/` is the handoff. Two details that are easy to get wrong:

**Out-of-fold is a training device only.** Saved aggregates use **full** training data with the same smoothing constants. Using OOF at inference discards 20% of the evidence; using full-data encoding during training reintroduces the leak.

**Count features must match the training distribution.** Training counts came from 4/5 of the data, so the app scales its full-data counts by `(k-1)/k`. Without it every `*_n` feature arrives ~25% larger than anything the model saw, and count-based split thresholds misfire — a silent accuracy drop rather than a crash.

**Cold start falls out of the smoothing for free.** An unknown key gives `(0 + 20 * GM) / (0 + 20) = GM`, the global mean — exactly the right prior when nothing is known. No special-case branch, so no special-case branch to get wrong.

---

## 9. Known limitations

1. **Accuracy is capped by data collection, not modelling.** 68% single-visit users. Cross-visit identity resolution is the highest-value fix.
2. **The catalogue is 30 attractions.** All are in Bali / Java. The recommender does not generalise to the full 1,698-attraction catalogue without transaction data for those items.
3. **Business class (1.2%, n=124) is near-unpredictable** — recall 0.30 even in the best model. A 34× imbalance against the majority class.
4. **The satisfaction threshold sweep was run on test.** ROC-AUC is unaffected, but the quoted accuracy-optimal threshold of 0.25 is mildly optimistic. A validation-split version is included in the notebook.
5. **Temporal validation was not used.** The split is random, not chronological. Transactions span 2013–2022; a time-based split would be a stricter test of deployment behaviour.
