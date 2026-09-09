
import os, joblib, numpy as np, pandas as pd, streamlit as st
import plotly.express as px

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🌍", layout="wide")
A = "artifacts"

# ---------------------------------------------------------------- load artifacts
@st.cache_resource
def load_models():
    return (joblib.load(f"{A}/visitmode_classifier.pkl"),
            joblib.load(f"{A}/rating_regressor.pkl"),
            joblib.load(f"{A}/satisfaction_classifier.pkl"),
            joblib.load(f"{A}/aggregates.pkl"),
            joblib.load(f"{A}/lookups.pkl"),
            joblib.load(f"{A}/recommender.pkl"),
            joblib.load(f"{A}/attr_country.pkl"))

@st.cache_data
def load_data():
    d = pd.read_csv(f"{A}/tourism_clean.csv")
    tables = {}
    for n in ("regression_results", "classification_results", "recommender_results"):
        p = f"{A}/{n}.csv"
        if os.path.exists(p):
            tables[n] = pd.read_csv(p)
    return d, tables

CLF, REG, SAT, AG, LK, REC, ATTR_COUNTRY = load_models()
df, TABLES = load_data()

CLASSES    = LK["classes"]
MODE_NAMES = {int(k): v for k, v in LK["mode_names"].items()}
GM         = AG["_global_mean"]
SCALE      = (LK["n_folds"] - 1) / LK["n_folds"]     # match Cell 6's test-side count scaling
ITEMS      = REC["items"]
I_POS      = {a: j for j, a in enumerate(ITEMS)}
META       = REC["meta"]
PRIOR      = np.array(AG["_prior"], dtype=float)

RATING_ENC = ["user_rt", "attr_rt", "country_rt", "city_rt", "type_rt", "cont_rt"]
MODE_ENC   = ["umode", "amode", "cmode", "citymode", "contmode", "tmode"]

# ---------------------------------------------------------------- feature builder
def encode(ctx, feats):
    """Reproduce Cell 6's encodings using FULL-TRAIN aggregates and the same smoothing."""
    r, mr = dict(ctx), AG["_smooth_r"]

    for out in RATING_ENC:
        spec = AG[out]; kv = ctx.get(spec["key"], -1)
        s = spec["sum"].get(kv, 0.0)
        c = spec["count"].get(kv, 0.0)
        r[out]          = (s + mr * GM) / (c + mr)      # unknown key -> global mean
        r[out + "_n"]   = c * SCALE

    for pre in MODE_ENC:
        spec = AG[pre]; kv = ctx.get(spec["key"], -1)
        m    = AG["_mode_smooth"][pre]
        cnts = spec["counts"].get(kv, {})
        vec  = np.array([cnts.get(c, 0.0) for c in CLASSES], dtype=float)
        n    = vec.sum()
        p    = (vec + m * PRIOR) / (n + m)
        for j, c in enumerate(CLASSES):
            r[f"{pre}_p{c}"] = p[j]
        r[f"{pre}_n"]   = n * SCALE
        r[f"{pre}_top"] = int(CLASSES[int(vec.argmax())]) if n > 0 else -1

    r["user_vs_global"] = r["user_rt"] - GM
    r["attr_vs_global"] = r["attr_rt"] - GM
    r["user_attr_gap"]  = r["user_rt"] - r["attr_rt"]
    return pd.DataFrame([{f: r.get(f, 0.0) for f in feats}])[feats]


def build_ctx(cont_id, country_id, city_id, attr_id, year, month, user_id=-1,
              rating=None, visit_mode=None):
    a = df[df.AttractionId == attr_id].iloc[0]
    ctx = dict(UserId=int(user_id), ContinentId=int(cont_id), CountryId=int(country_id),
               CityId=int(city_id), AttractionId=int(attr_id),
               AttractionTypeId=int(a.AttractionTypeId),
               AttractionCityId=int(a.AttractionCityId),
               RegionId=int(df[df.CountryId == country_id].iloc[0].RegionId),
               VisitYear=int(year), VisitMonth=int(month),
               VisitQuarter=(int(month) - 1) // 3 + 1,
               MonthSin=float(np.sin(2 * np.pi * month / 12)),
               MonthCos=float(np.cos(2 * np.pi * month / 12)),
               IsDomestic=int(ATTR_COUNTRY.get(int(attr_id), -1) == int(country_id)))
    if rating is not None:
        ctx["Rating"] = int(rating)
    if visit_mode is not None:
        ctx["VisitMode"] = int(visit_mode)
    return ctx


def recommend(cont_id, exclude=(), k=5):
    cl = REC["cont_logp"]
    s = (cl.loc[cont_id].to_numpy(float).copy() if cont_id in cl.index
         else np.array(REC["global_logp"], dtype=float).copy())
    for a in exclude:
        if a in I_POS:
            s[I_POS[a]] = -np.inf
    idx = np.argsort(-s)[:k]
    ids = np.array(ITEMS)[idx]
    return pd.DataFrame({
        "Attraction": META["Attraction"].reindex(ids).values,
        "Type"      : META["AttractionType"].reindex(ids).values,
        "Avg rating": [round(REC["item_mean"].get(int(i), np.nan), 2) for i in ids],
        "Visits"    : [REC["item_visits"].get(int(i), 0) for i in ids],
        "Affinity"  : np.round(s[idx], 3)})

# ---------------------------------------------------------------- UI
st.title("🌍 Tourism Experience Analytics")
st.caption(f"Visit mode: {LK['best_clf']}  ·  Rating: {LK['best_reg']}  ·  "
           f"Recommender: {LK['best_rec']}")

t1, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "🎯 Visit Mode", "⭐ Rating & Satisfaction",
                              "💡 Recommendations", "📈 Model Performance"])

# ---- Dashboard ----
with t1:
    c = st.columns(4)
    c[0].metric("Transactions", f"{len(df):,}")
    c[1].metric("Users", f"{df.UserId.nunique():,}")
    c[2].metric("Attractions", f"{df.AttractionId.nunique():,}")
    c[3].metric("Avg rating", f"{df.Rating.mean():.2f} ⭐")

    f1, f2 = st.columns(2)
    conts = f1.multiselect("Continent", sorted(df.Continent.unique()),
                           sorted(df.Continent.unique()))
    yrs = f2.slider("Visit year", int(df.VisitYear.min()), int(df.VisitYear.max()),
                    (int(df.VisitYear.min()), int(df.VisitYear.max())))
    d = df[df.Continent.isin(conts) & df.VisitYear.between(*yrs)]
    if d.empty:
        st.warning("No rows match those filters.")
    else:
        a, b = st.columns(2)
        a.plotly_chart(px.bar(d.VisitModeName.value_counts().reset_index(),
                              x="VisitModeName", y="count", color="VisitModeName",
                              title="Visit modes"), use_container_width=True)
        b.plotly_chart(px.pie(d.Continent.value_counts().reset_index(), names="Continent",
                              values="count", hole=.45, title="Visitors by continent"),
                       use_container_width=True)
        top = (d.groupby("Attraction")
                 .agg(Visits=("Rating", "size"), AvgRating=("Rating", "mean"))
                 .sort_values("Visits", ascending=False).head(15).reset_index())
        st.plotly_chart(px.bar(top, x="Visits", y="Attraction", color="AvgRating",
                               orientation="h", color_continuous_scale="RdYlGn", height=520,
                               title="Top attractions (colour = average rating)"),
                        use_container_width=True)
        st.plotly_chart(px.line(d.groupby(["VisitYear", "VisitModeName"]).size()
                                 .reset_index(name="n"),
                                x="VisitYear", y="n", color="VisitModeName", markers=True,
                                title="Visit-mode trend"), use_container_width=True)

# ---- shared input widget ----
def traveller_inputs(key):
    c1, c2, c3 = st.columns(3)
    cont = c1.selectbox("Continent", sorted(df.Continent.unique()), key=f"{key}_cont")
    ctry = c2.selectbox("Country", sorted(df[df.Continent == cont].Country.unique()),
                        key=f"{key}_ctry")
    cityn = c3.selectbox("Home city", sorted(df[df.Country == ctry].UserCity.unique()),
                         key=f"{key}_city")
    c4, c5, c6 = st.columns(3)
    attr = c4.selectbox("Attraction", sorted(df.Attraction.unique()), key=f"{key}_attr")
    year = c5.number_input("Visit year", 2013, 2030, 2022, key=f"{key}_yr")
    month = c6.slider("Visit month", 1, 12, 7, key=f"{key}_mo")
    return (int(df[df.Continent == cont].iloc[0].ContinentId),
            int(df[df.Country == ctry].iloc[0].CountryId),
            int(df[df.UserCity == cityn].iloc[0].CityId),
            int(df[df.Attraction == attr].iloc[0].AttractionId),
            year, month, cont)

# ---- Visit mode ----
with t2:
    st.subheader("Predict how this traveller will visit")
    cid, coid, ciid, aid, yr, mo, cont_name = traveller_inputs("vm")
    rt = st.slider("Expected rating", 1, 5, 4, key="vm_rt")
    if st.button("Predict visit mode", type="primary"):
        X = encode(build_ctx(cid, coid, ciid, aid, yr, mo, rating=rt), LK["clf_feats"])
        pr = CLF.predict_proba(X)[0]
        st.success(f"Predicted: **{MODE_NAMES[CLASSES[int(pr.argmax())]]}** "
                   f"({pr.max()*100:.1f}% confidence)")
        st.plotly_chart(px.bar(x=[MODE_NAMES[c] for c in CLASSES], y=pr,
                               labels={"x": "Visit mode", "y": "Probability"},
                               title="Class probabilities"), use_container_width=True)
        st.caption("Model accuracy 62.9% vs a 40.9% majority baseline; top-2 accuracy 81.6%. "
                   "New users have no visit history, which is the main limit on accuracy.")

# ---- Rating & satisfaction ----
with t3:
    st.subheader("Predict the rating and flag dissatisfaction risk")
    cid, coid, ciid, aid, yr, mo, cont_name = traveller_inputs("rt")
    vm = st.selectbox("Visit mode", CLASSES, format_func=lambda c: MODE_NAMES[c], key="rt_vm")
    if st.button("Predict rating", type="primary"):
        ctx = build_ctx(cid, coid, ciid, aid, yr, mo, visit_mode=vm)
        yhat = float(np.clip(REG.predict(encode(ctx, LK["reg_feats"]))[0], 1, 5))
        risk = float(SAT.predict_proba(encode(ctx, LK["reg_feats"]))[0, 0])
        a, b = st.columns(2)
        a.metric("Predicted rating", f"{yhat:.2f} / 5", f"{yhat - df.Rating.mean():+.2f} vs average")
        b.metric("Dissatisfaction risk", f"{risk*100:.1f}%",
                 "flagged" if risk >= LK["sat_threshold"] else "not flagged")
        st.progress(yhat / 5)
        if risk >= LK["sat_threshold"]:
            st.warning("⚠️ Route to service recovery. At this threshold the model catches "
                       "61% of dissatisfied visits at 34% precision (base rate 21%).")
        else:
            st.success("✅ Low dissatisfaction risk.")

# ---- Recommendations ----
with t4:
    st.subheader("Personalised attraction recommendations")
    st.info(f"Best model: **{LK['best_rec']}** — HitRate@5 49.8% vs 16.7% random. "
            "Collaborative filtering was tested and scored at chance level: only 30 "
            "attractions exist and users average 1.3 ratings each.")
    mode = st.radio("Traveller", ["New visitor", "Returning visitor"], horizontal=True)
    k = st.slider("How many?", 3, 10, 5)
    if mode == "New visitor":
        cont = st.selectbox("Continent of origin", sorted(df.Continent.unique()), key="rc_cont")
        cid = int(df[df.Continent == cont].iloc[0].ContinentId)
        if st.button("Recommend", type="primary"):
            st.dataframe(recommend(cid, k=k), use_container_width=True, hide_index=True)
    else:
        uid = st.selectbox("User ID", sorted(df.UserId.unique())[:3000], key="rc_uid")
        if st.button("Recommend", type="primary"):
            hist = df[df.UserId == uid]
            cid = int(hist.iloc[0].ContinentId)
            st.write("**Already visited**")
            st.dataframe(hist.groupby(["Attraction", "AttractionType"])["Rating"]
                             .agg(["size", "mean"]).round(2).reset_index(),
                         use_container_width=True, hide_index=True)
            st.write("**Recommended next**")
            st.dataframe(recommend(cid, exclude=set(hist.AttractionId), k=k),
                         use_container_width=True, hide_index=True)

# ---- Model performance ----
with t5:
    st.subheader("Model comparison")
    for name, title in [("classification_results", "Visit-mode classification"),
                        ("regression_results", "Rating regression"),
                        ("recommender_results", "Recommendation")]:
        if name in TABLES:
            st.markdown(f"**{title}**")
            st.dataframe(TABLES[name].round(4), use_container_width=True, hide_index=True)
    st.markdown("""
**Key findings**

1. **68% of users appear exactly once.** Visit-mode accuracy is capped near 63%
   (51% of rows have user history at 75% accuracy, 49% have none at ~47%). Six
   independent model families all landed within 0.9 points of that figure, so this
   is an information limit, not a tuning limit. *Cross-visit user identity tracking
   is worth more than any model change.*
2. **Collaborative filtering fails on this catalogue.** 30 attractions, 4.3% matrix
   density, 1.3 ratings per user. Damped CF scored 0.167 against a 0.167 random
   baseline. Popularity conditioned on continent is the correct production choice.
3. **Rating is driven by which attraction (corr 0.30) more than by who rates it
   (corr 0.17).** Service quality, not visitor mix, determines satisfaction.
""")
