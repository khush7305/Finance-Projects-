import warnings
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
AGE_REF_DATE = pd.Timestamp("2026-01-01")

VITAL_PATTERNS = {
    "weight": r"(?i)body weight|weight",
    "height": r"(?i)body height|height",
    "bmi": r"(?i)body mass index|bmi",
    "dbp": r"(?i)diastolic blood pressure|diastolic",
    "sbp": r"(?i)systolic blood pressure|systolic",
    "heart_rate": r"(?i)heart rate|pulse",
    "resp_rate": r"(?i)respiratory rate",
}

CHRONIC_REGEX = r"(?i)hypertension|diabetes|obesity|prediabetes|anemia|ischemic heart"

TARGET_REGEX_MAP = {
    "has_chronic": CHRONIC_REGEX,
    "has_hypertension": r"(?i)hypertension",
    "has_diabetes": r"(?i)diabetes|prediabetes",
    "has_obesity": r"(?i)obesity",
}

TARGET_LABEL_MAP = {
    "has_chronic": "Metabolic & Chronic Risk (Composite)",
    "has_hypertension": "Essential Hypertension",
    "has_diabetes": "Diabetes / Prediabetes",
    "has_obesity": "Obesity (BMI 30+)",
}


def style_plot(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    return ax


def inject_css():
    st.markdown(
        """
        <style>
        :root {
          --accent: #2563EB;
          --success: #16A34A;
          --warning: #D97706;
          --danger: #DC2626;
          --muted: #6B7280;
          --card-bg: #F9FAFB;
        }
        .metric-card {
          background: var(--card-bg);
          border-left: 6px solid var(--accent);
          border-radius: 10px;
          padding: 14px 16px;
          margin: 6px 0;
          color: #111111 !important;
        }
        .metric-title {
          color: #374151 !important;
          font-size: 0.9rem;
          margin: 0;
        }
        .metric-value {
          font-size: 1.2rem;
          font-weight: 700;
          margin: 2px 0 0 0;
          color: #111111 !important;
        }
        .info-box {
          background: #EFF6FF;
          border: 1px solid #BFDBFE;
          border-radius: 10px;
          padding: 12px 14px;
          margin: 8px 0;
          color: #111111 !important;
        }
        .warn-box {
          background: #FFFBEB;
          border: 1px solid #FCD34D;
          border-radius: 10px;
          padding: 12px 14px;
          margin: 8px 0;
          color: #111111 !important;
        }
        .metric-card *,
        .info-box *,
        .warn-box * {
          color: #111111 !important;
        }
        .footer {
          text-align: center;
          color: var(--muted);
          font-size: 0.9rem;
          margin-top: 24px;
          padding-top: 14px;
          border-top: 1px solid #E5E7EB;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
          <p class="metric-title">{title}</p>
          <p class="metric-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)


@st.cache_data(show_spinner=False)
def load_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    base = Path(data_dir)
    files = {
        "patients": "patients.csv",
        "encounters": "encounters.csv",
        "conditions": "conditions.csv",
        "observations": "observations.csv",
        "medications": "medications.csv",
    }
    out = {}
    for key, name in files.items():
        path = base / name
        try:
            out[key] = pd.read_csv(path, low_memory=False)
        except Exception:
            out[key] = pd.read_csv(path, engine="python", on_bad_lines="skip")
    return out


def build_target(
    patients: pd.DataFrame, conditions: pd.DataFrame, target_key: str
) -> pd.DataFrame:
    c = conditions[["PATIENT", "DESCRIPTION"]].copy()
    c["DESCRIPTION"] = c["DESCRIPTION"].fillna("")
    pattern = TARGET_REGEX_MAP.get(target_key, CHRONIC_REGEX)
    chronic_pos = c[c["DESCRIPTION"].str.contains(pattern, regex=True, na=False)][
        "PATIENT"
    ].drop_duplicates()
    y = pd.DataFrame({"PATIENT": patients["Id"].copy()})
    y["has_chronic"] = y["PATIENT"].isin(chronic_pos).astype(int)
    return y


def build_observation_features(observations: pd.DataFrame) -> pd.DataFrame:
    obs = observations[["PATIENT", "DATE", "DESCRIPTION", "VALUE"]].copy()
    obs["DATE"] = safe_datetime(obs["DATE"])
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs = obs.dropna(subset=["DATE", "VALUE"])

    obs["vital"] = None
    for vital, pattern in VITAL_PATTERNS.items():
        mask = obs["DESCRIPTION"].str.contains(pattern, regex=True, na=False)
        obs.loc[mask, "vital"] = vital
    obs = obs.dropna(subset=["vital"])
    split_date = obs["DATE"].quantile(0.70)
    obs["period"] = np.where(obs["DATE"] < split_date, "DS1", "DS2")

    agg = (
        obs.groupby(["PATIENT", "period", "vital"])["VALUE"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)
    wide = agg.pivot_table(
        index=["PATIENT", "period"], columns="vital", values=["mean", "std"]
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    trend = (
        obs.sort_values("DATE")
        .groupby(["PATIENT", "period", "vital"])
        .agg(
            first_value=("VALUE", "first"),
            last_value=("VALUE", "last"),
            first_date=("DATE", "first"),
            last_date=("DATE", "last"),
            n_obs=("VALUE", "count"),
        )
        .reset_index()
    )
    trend["time_days"] = (
        (trend["last_date"] - trend["first_date"]).dt.days.clip(lower=1).fillna(1)
    )
    trend["trend_delta"] = trend["last_value"] - trend["first_value"]
    trend["trend_per_day"] = trend["trend_delta"] / trend["time_days"]
    trend_wide = trend.pivot_table(
        index=["PATIENT", "period"],
        columns="vital",
        values=["trend_delta", "trend_per_day", "n_obs"],
    )
    trend_wide.columns = [f"{a}_{b}" for a, b in trend_wide.columns]
    anchors = (
        obs.groupby(["PATIENT", "period"])["DATE"]
        .max()
        .reset_index(name="obs_anchor_date")
    )
    out = pd.merge(
        wide.reset_index(), trend_wide.reset_index(), on=["PATIENT", "period"], how="left"
    )
    return pd.merge(out, anchors, on=["PATIENT", "period"], how="left")


def build_count_features(
    encounters: pd.DataFrame, medications: pd.DataFrame
) -> pd.DataFrame:
    e = encounters[["PATIENT", "START"]].copy()
    e["START"] = safe_datetime(e["START"])
    e = e.dropna(subset=["START"])
    split_date_e = e["START"].quantile(0.70)
    e["period"] = np.where(e["START"] < split_date_e, "DS1", "DS2")
    e_cnt = e.groupby(["PATIENT", "period"]).size().reset_index(name="encounter_count")
    e_anchor = (
        e.groupby(["PATIENT", "period"])["START"].max().reset_index(name="enc_anchor_date")
    )

    m = medications[["PATIENT", "START"]].copy()
    m["START"] = safe_datetime(m["START"])
    m = m.dropna(subset=["START"])
    split_date_m = m["START"].quantile(0.70)
    m["period"] = np.where(m["START"] < split_date_m, "DS1", "DS2")
    m_cnt = m.groupby(["PATIENT", "period"]).size().reset_index(name="medication_count")
    m_anchor = (
        m.groupby(["PATIENT", "period"])["START"].max().reset_index(name="med_anchor_date")
    )

    out = pd.merge(e_cnt, m_cnt, on=["PATIENT", "period"], how="outer")
    out = pd.merge(out, e_anchor, on=["PATIENT", "period"], how="left")
    out = pd.merge(out, m_anchor, on=["PATIENT", "period"], how="left")
    out["encounter_count"] = out["encounter_count"].fillna(0)
    out["medication_count"] = out["medication_count"].fillna(0)
    return out


def build_demographics(patients: pd.DataFrame) -> pd.DataFrame:
    p = patients[
        [
            "Id",
            "BIRTHDATE",
            "GENDER",
            "RACE",
            "ETHNICITY",
            "INCOME",
            "HEALTHCARE_EXPENSES",
            "HEALTHCARE_COVERAGE",
        ]
    ].copy()
    p = p.rename(columns={"Id": "PATIENT"})
    p["BIRTHDATE"] = safe_datetime(p["BIRTHDATE"])
    p["age"] = ((AGE_REF_DATE - p["BIRTHDATE"]).dt.days / 365.25).clip(lower=0)
    p["gender_bin"] = p["GENDER"].map({"M": 1, "F": 0}).fillna(0)

    p["RACE"] = p["RACE"].fillna("Unknown")
    p["ETHNICITY"] = p["ETHNICITY"].fillna("Unknown")
    dummies = pd.get_dummies(p[["RACE", "ETHNICITY"]], prefix=["RACE", "ETH"])

    num = p[
        [
            "PATIENT",
            "age",
            "gender_bin",
            "INCOME",
            "HEALTHCARE_EXPENSES",
            "HEALTHCARE_COVERAGE",
        ]
    ].copy()
    return pd.concat([num, dummies], axis=1)


@st.cache_data(show_spinner=True)
def build_datasets(data_dir: str, target_key: str):
    tables = load_data(data_dir)
    patients = tables["patients"]
    encounters = tables["encounters"]
    conditions = tables["conditions"]
    observations = tables["observations"]
    medications = tables["medications"]

    target = build_target(patients, conditions, target_key=target_key)
    obs_features = build_observation_features(observations)
    count_features = build_count_features(encounters, medications)
    demo = build_demographics(patients)

    base = pd.merge(obs_features, count_features, on=["PATIENT", "period"], how="outer")
    base = pd.merge(base, demo, on="PATIENT", how="left")
    base = pd.merge(base, target, on="PATIENT", how="left")
    base = base[base["has_chronic"].notna()].copy()
    anchor_cols = ["obs_anchor_date", "enc_anchor_date", "med_anchor_date"]
    for col in anchor_cols:
        base[col] = pd.to_datetime(base[col], errors="coerce")
    base["anchor_date"] = base[anchor_cols].max(axis=1)
    global_anchor = base["anchor_date"].dropna().max()
    base["anchor_date"] = base["anchor_date"].fillna(global_anchor)

    ds1 = base[base["period"] == "DS1"].copy()
    ds2 = base[base["period"] == "DS2"].copy()

    feature_cols = [
        c
        for c in base.columns
        if c
        not in [
            "PATIENT",
            "period",
            "has_chronic",
            "obs_anchor_date",
            "enc_anchor_date",
            "med_anchor_date",
            "anchor_date",
            "encounter_count",
            "medication_count",
            "HEALTHCARE_EXPENSES",
            "HEALTHCARE_COVERAGE",
        ]
    ]
    ds1 = ds1[["PATIENT", "period", "has_chronic"] + feature_cols]
    ds2 = ds2[["PATIENT", "period", "has_chronic", "anchor_date"] + feature_cols]
    return ds1, ds2, feature_cols


def oversample_minority(X: pd.DataFrame, y: pd.Series, ratio: int = 5):
    data = X.copy()
    data["target"] = y.values
    pos = data[data["target"] == 1]
    if pos.empty:
        return X, y
    extra = pd.concat([pos] * ratio, ignore_index=True)
    combo = pd.concat([data, extra], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )
    return combo.drop(columns=["target"]), combo["target"]


def safe_stratify_split(X: pd.DataFrame, y: pd.Series):
    if y.nunique() < 2 or y.value_counts().min() < 2:
        return train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
        )
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def temporal_split_by_anchor(
    X: pd.DataFrame, y: pd.Series, anchor_dates: pd.Series, test_size: float = 0.2
):
    split_df = pd.DataFrame(
        {"idx": np.arange(len(X)), "y": y.values, "anchor": anchor_dates.values}
    )
    split_df["anchor"] = pd.to_datetime(split_df["anchor"], errors="coerce")
    split_df = split_df.sort_values("anchor")
    n_test = max(1, int(len(split_df) * test_size))
    test_idx = split_df.iloc[-n_test:]["idx"].values
    train_idx = split_df.iloc[:-n_test]["idx"].values

    # Fallback if chronological holdout accidentally leaves a single class.
    if len(np.unique(y.iloc[test_idx])) < 2 or len(train_idx) < 20:
        return safe_stratify_split(X, y)

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def make_pipeline(model):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )


def get_scores(y_true, y_pred, y_prob):
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred),
    }
    if len(np.unique(y_true)) > 1 and y_prob is not None:
        scores["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        scores["AUC"] = np.nan
    return scores


def optimize_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if y_prob is None or len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.linspace(0.2, 0.8, 31)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def predict_with_threshold(model, X: pd.DataFrame, threshold: float):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        return y_pred, y_prob
    y_pred = model.predict(X)
    return y_pred, None


@st.cache_resource(show_spinner=True)
def train_all_models(data_dir: str, target_key: str):
    ds1, ds2, feature_cols = build_datasets(data_dir, target_key=target_key)
    X1, y1 = ds1[feature_cols], ds1["has_chronic"].astype(int)
    X2, y2 = ds2[feature_cols], ds2["has_chronic"].astype(int)

    X1_train, X1_test, y1_train, y1_test = safe_stratify_split(X1, y1)
    X2_train, X2_test, y2_train, y2_test = temporal_split_by_anchor(
        X2, y2, ds2["anchor_date"], test_size=0.2
    )

    X1_train_os, y1_train_os = oversample_minority(X1_train, y1_train, ratio=5)
    X2_train_os, y2_train_os = oversample_minority(X2_train, y2_train, ratio=5)

    models = {
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "SVM": make_pipeline(
            SVC(
                C=10,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "MLP": make_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=600,
                alpha=0.01,
                random_state=RANDOM_STATE,
            )
        ),
    }

    metrics = []
    curves_ds1 = {}
    curves_ds2 = {}
    trained = {}
    model_thresholds = {}

    for name, pipe in models.items():
        pipe.fit(X1_train_os, y1_train_os)
        trained[name] = pipe
        y_prob_ds1_test = (
            pipe.predict_proba(X1_test)[:, 1]
            if hasattr(pipe, "predict_proba")
            else None
        )
        best_thr = optimize_threshold(y1_test, y_prob_ds1_test)
        model_thresholds[name] = best_thr

        for split_name, Xs, ys in [
            ("DS1_Train", X1_train, y1_train),
            ("DS1_Test", X1_test, y1_test),
            ("DS2_Test", X2_test, y2_test),
        ]:
            yp, yp_prob = predict_with_threshold(pipe, Xs, best_thr)
            s = get_scores(ys, yp, yp_prob)
            row = {"Model": name, "Split": split_name}
            row.update({k: v for k, v in s.items() if k != "Confusion_Matrix"})
            row["Confusion_Matrix"] = s["Confusion_Matrix"]
            row["Threshold"] = best_thr
            metrics.append(row)

            if yp_prob is not None and len(np.unique(ys)) > 1:
                fpr, tpr, _ = roc_curve(ys, yp_prob)
                if split_name == "DS1_Test":
                    curves_ds1[name] = (fpr, tpr)
                if split_name == "DS2_Test":
                    curves_ds2[name] = (fpr, tpr)

    metrics_df = pd.DataFrame(metrics)

    # Continual learning: MLP warm-start + DS2 threshold optimization
    mlp_cl = copy.deepcopy(trained["MLP"])
    mlp_cl.named_steps["clf"].warm_start = True
    mlp_cl.named_steps["clf"].max_iter = 500
    mlp_cl.named_steps["clf"].alpha = 0.005
    X2_train_t = mlp_cl.named_steps["scaler"].transform(
        mlp_cl.named_steps["imputer"].transform(X2_train_os)
    )
    mlp_cl.named_steps["clf"].fit(X2_train_t, y2_train_os)

    # Continual learning: Decision tree replay with DS2 emphasis
    dt_replay = clone(trained["Decision Tree"])
    X_replay = pd.concat([X1_train_os, X2_train_os, X2_train_os], ignore_index=True)
    y_replay = pd.concat([y1_train_os, y2_train_os, y2_train_os], ignore_index=True)
    dt_replay.fit(X_replay, y_replay)

    # Baseline vs CL metrics on DS2 test
    cl_rows = []
    cl_thresholds = {
        "Baseline_DT_DS1->DS2": model_thresholds["Decision Tree"],
        "Baseline_MLP_DS1->DS2": model_thresholds["MLP"],
    }
    dt_replay_prob_train = dt_replay.predict_proba(X2_train)[:, 1]
    cl_thresholds["Replay_DT_DS1+DS2"] = optimize_threshold(y2_train, dt_replay_prob_train)
    mlp_cl_prob_train = mlp_cl.predict_proba(X2_train)[:, 1]
    cl_thresholds["WarmStart_MLP_DS2_FT"] = optimize_threshold(y2_train, mlp_cl_prob_train)

    for label, model in [
        ("Baseline_DT_DS1->DS2", trained["Decision Tree"]),
        ("Baseline_MLP_DS1->DS2", trained["MLP"]),
        ("Replay_DT_DS1+DS2", dt_replay),
        ("WarmStart_MLP_DS2_FT", mlp_cl),
    ]:
        y_pred, y_prob = predict_with_threshold(model, X2_test, cl_thresholds[label])
        s = get_scores(y2_test, y_pred, y_prob)
        cl_rows.append(
            {
                "Model_Variant": label,
                "Accuracy": s["Accuracy"],
                "Precision": s["Precision"],
                "Recall": s["Recall"],
                "F1": s["F1"],
                "AUC": s["AUC"],
                "Threshold": cl_thresholds[label],
            }
        )
    cl_df = pd.DataFrame(cl_rows)

    # Bias-variance proxy table
    bv = (
        metrics_df[metrics_df["Split"].isin(["DS1_Train", "DS1_Test"])]
        .pivot(index="Model", columns="Split", values=["Accuracy", "F1"])
        .reset_index()
    )
    bv.columns = [
        "Model",
        "Accuracy_DS1_Train",
        "Accuracy_DS1_Test",
        "F1_DS1_Train",
        "F1_DS1_Test",
    ]
    bv["Accuracy_Gap"] = bv["Accuracy_DS1_Train"] - bv["Accuracy_DS1_Test"]
    bv["F1_Gap"] = bv["F1_DS1_Train"] - bv["F1_DS1_Test"]

    return {
        "ds1": ds1,
        "ds2": ds2,
        "target_key": target_key,
        "target_label": TARGET_LABEL_MAP.get(target_key, target_key),
        "feature_cols": feature_cols,
        "splits": (
            X1_train,
            X1_test,
            y1_train,
            y1_test,
            X2_train,
            X2_test,
            y2_train,
            y2_test,
        ),
        "trained": trained,
        "metrics_df": metrics_df,
        "curves_ds1": curves_ds1,
        "curves_ds2": curves_ds2,
        "cl_models": {"mlp_warmstart": mlp_cl, "dt_replay": dt_replay},
        "cl_df": cl_df,
        "bias_variance_df": bv,
        "model_thresholds": model_thresholds,
    }


@st.cache_data(show_spinner=False)
def build_dataset_relational_map(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    csv_files = sorted(base.glob("*.csv"))
    tables = {}
    for fp in csv_files:
        try:
            tables[fp.stem] = pd.read_csv(fp, low_memory=False)
        except Exception:
            tables[fp.stem] = pd.read_csv(fp, engine="python", on_bad_lines="skip")

    key_like = {"Id", "PATIENT", "ENCOUNTER", "PAYER", "PROVIDER", "ORGANIZATION", "CLAIMID"}
    edges = []
    summary = []
    for name, df in tables.items():
        cols = set(df.columns)
        summary.append(
            {
                "Table": name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "ID_Columns": ", ".join(sorted([c for c in cols if c in key_like])) or "-",
            }
        )

    names = list(tables.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ca = set(tables[a].columns)
            cb = set(tables[b].columns)
            shared = sorted((ca & cb) & key_like)
            if not shared:
                continue
            best_overlap = 0.0
            best_key = shared[0]
            for key in shared:
                va = set(tables[a][key].dropna().astype(str).unique()) if key in tables[a].columns else set()
                vb = set(tables[b][key].dropna().astype(str).unique()) if key in tables[b].columns else set()
                if not va or not vb:
                    continue
                overlap = len(va & vb) / max(1, min(len(va), len(vb)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_key = key
            edges.append(
                {
                    "From": a,
                    "To": b,
                    "Join_Key": best_key,
                    "Overlap_Ratio": best_overlap,
                }
            )
    return pd.DataFrame(summary), pd.DataFrame(edges).sort_values("Overlap_Ratio", ascending=False)


def plot_class_distribution(df: pd.DataFrame, title: str):
    counts = df["has_chronic"].value_counts().sort_index()
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(
        ["0 (No Chronic)", "1 (Chronic)"], counts.values, color=["#2563EB", "#DC2626"]
    )
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{pct.iloc[i]:.1f}%",
            ha="center",
            va="bottom",
        )
    ax.set_title(title)
    style_plot(ax)
    st.pyplot(fig)


def plot_missing(df: pd.DataFrame, n: int = 12):
    miss = df.isna().mean().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#D97706")
    ax.set_xlabel("Missing Ratio")
    ax.set_ylabel("")
    style_plot(ax)
    st.pyplot(fig)


def plot_hist_grid(df: pd.DataFrame, title_prefix: str):
    candidates = [
        "mean_weight",
        "mean_height",
        "mean_bmi",
        "mean_dbp",
        "mean_sbp",
        "mean_heart_rate",
    ]
    cols = [c for c in candidates if c in df.columns][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(
            data=df, x=col, hue="has_chronic", ax=axes[i], bins=25, kde=True, alpha=0.45
        )
        axes[i].set_title(f"{title_prefix}: {col}")
        style_plot(axes[i])
    for j in range(len(cols), 6):
        axes[j].axis("off")
    plt.tight_layout()
    st.pyplot(fig)


def plot_corr_heatmap(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.drop(
        columns=[c for c in ["has_chronic"] if c in num.columns], errors="ignore"
    )
    if num.shape[1] < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return
    corr = num.corr().fillna(0)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar=True)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)


def plot_confusion_pair(cm_train, cm_test, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{title} - DS1 Train")
    axes[0].set_xlabel("Pred")
    axes[0].set_ylabel("True")
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Purples", ax=axes[1])
    axes[1].set_title(f"{title} - DS1 Test")
    axes[1].set_xlabel("Pred")
    axes[1].set_ylabel("True")
    st.pyplot(fig)


def sidebar():
    with st.sidebar:
        
        st.title("Assignment-2 ML Dashboard")
        st.caption("BITS F464 Assignment 2")
        #st.caption("Temporal Shift in EHR")
        default_dir = str((Path(__file__).resolve().parent / "data"))
        data_dir = st.text_input("Data directory", value=default_dir)
        target_key = st.selectbox(
            "Target variable",
            options=list(TARGET_REGEX_MAP.keys()),
            format_func=lambda x: TARGET_LABEL_MAP.get(x, x),
            index=0,
        )
        page = st.radio(
            "Navigate",
            [
                "🏠 Overview",
                "🕸 Dataset Relational Map",
                "📊 EDA — Dataset 1",
                "📊 EDA — Dataset 2",
                "🤖 Model Training & Evaluation",
                "📉 Temporal Shift Analysis",
                "🔄 Continual Learning",
                "🌳 Feature Importance & Interpretation",
            ],
        )
    return data_dir, target_key, page


def resolve_data_dir(data_dir: str) -> Path:
    p = Path(data_dir)
    required = [
        "patients.csv",
        "encounters.csv",
        "conditions.csv",
        "observations.csv",
        "medications.csv",
    ]
    if p.exists() and all((p / f).exists() for f in required):
        return p
    fallback = p / "data"
    if fallback.exists() and all((fallback / f).exists() for f in required):
        return fallback
    return p


def render_overview(results):
    ds1, ds2 = results["ds1"], results["ds2"]
    metrics_df = results["metrics_df"]
    target_label = results.get("target_label", "Any Chronic Condition")

    st.markdown(
        """
        <div class="info-box">
        <b>Pipeline Summary:</b> Multi-table integration → temporal split (DS1/DS2) → chronic target creation →
        engineered vitals/demographic/count features → model training on DS1 → cross-time evaluation on DS2 →
        continual learning with warm-start and replay.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(5)
    steps = [
        ("Step 1", "Load 5 EHR CSV tables"),
        ("Step 2", "Engineer vitals + usage + demographics"),
        ("Step 3", "Create chronic binary target"),
        ("Step 4", "Train DT / SVM / MLP on DS1"),
        ("Step 5", "Evaluate drift + continual learning"),
    ]
    for col, (t, v) in zip(cards, steps):
        with col:
            render_metric_card(t, v)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DS1 Rows", f"{len(ds1):,}")
    c2.metric("DS2 Rows", f"{len(ds2):,}")
    c3.metric("DS1 Positive Rate", f"{ds1['has_chronic'].mean() * 100:.2f}%")
    c4.metric("DS2 Positive Rate", f"{ds2['has_chronic'].mean() * 100:.2f}%")
    st.caption(f"Active target: {target_label}")

    ds1_test = metrics_df[metrics_df["Split"] == "DS1_Test"].copy()
    show_cols = [
        "Model",
        "Accuracy",
        "Balanced_Acc",
        "Precision",
        "Recall",
        "F1",
        "Macro_F1",
        "AUC",
    ]
    st.dataframe(
        ds1_test[show_cols].sort_values("AUC", ascending=False),
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="warn-box">
        Class imbalance is severe (often near 97% negative in many healthcare cohorts). Metrics like
        Balanced Accuracy, Recall, F1, and AUC are more informative than plain Accuracy alone.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_eda(ds: pd.DataFrame, title: str):
    st.subheader(title)
    st.write(f"Shape: **{ds.shape[0]} rows × {ds.shape[1]} columns**")

    c1, c2 = st.columns(2)
    with c1:
        plot_class_distribution(ds, f"{title} - Target Distribution")
    with c2:
        plot_missing(ds)

    plot_hist_grid(ds, title)
    st.markdown("**Descriptive Statistics**")
    st.dataframe(
        ds.describe(include="all").transpose().head(30), use_container_width=True
    )
    plot_corr_heatmap(ds)


def render_model_training(results):
    metrics_df = results["metrics_df"]
    curves_ds1 = results["curves_ds1"]
    bv = results["bias_variance_df"]
    model_thresholds = results["model_thresholds"]

    model_name = st.selectbox("Select model", ["Decision Tree", "SVM", "MLP"])
    m = metrics_df[metrics_df["Model"] == model_name]
    train_row = m[m["Split"] == "DS1_Train"].iloc[0]
    test_row = m[m["Split"] == "DS1_Test"].iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**DS1 Train Metrics**")
        st.metric("Accuracy", f"{train_row['Accuracy']:.3f}")
        st.metric("Precision", f"{train_row['Precision']:.3f}")
        st.metric("Recall", f"{train_row['Recall']:.3f}")
        st.metric("F1", f"{train_row['F1']:.3f}")
    with c2:
        st.markdown("**DS1 Test Metrics**")
        st.metric("Accuracy", f"{test_row['Accuracy']:.3f}")
        st.metric("Precision", f"{test_row['Precision']:.3f}")
        st.metric("Recall", f"{test_row['Recall']:.3f}")
        st.metric("F1", f"{test_row['F1']:.3f}")
    st.caption(
        f"Decision threshold for {model_name}: {model_thresholds.get(model_name, 0.5):.2f}"
    )

    plot_confusion_pair(
        train_row["Confusion_Matrix"], test_row["Confusion_Matrix"], model_name
    )

    st.markdown("**ROC Curves (DS1 Test, all models)**")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, (fpr, tpr) in curves_ds1.items():
        auc = metrics_df[
            (metrics_df["Model"] == name) & (metrics_df["Split"] == "DS1_Test")
        ]["AUC"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    style_plot(ax)
    st.pyplot(fig)

    st.markdown("**Model Comparison (DS1 Test)**")
    comp = metrics_df[metrics_df["Split"] == "DS1_Test"][
        ["Model", "Accuracy", "Precision", "Recall", "F1"]
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    for i, met in enumerate(metric_names):
        sns.barplot(data=comp, x="Model", y=met, ax=axes[i], palette="Blues_r")
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis="x", rotation=20)
        axes[i].set_title(met)
        style_plot(axes[i])
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Bias-Variance Analysis (Train vs Test Gap)**")
    st.dataframe(bv, use_container_width=True)
    st.markdown(
        "- Large train-test gap suggests overfitting.\n"
        "- Low train and test scores suggest underfitting.\n"
        "- Smaller gaps with strong test scores indicate better generalization."
    )


def render_temporal_shift(results):
    metrics_df = results["metrics_df"]
    curves_ds2 = results["curves_ds2"]
    ds1, ds2 = results["ds1"], results["ds2"]

    auc_comp = metrics_df[metrics_df["Split"].isin(["DS1_Test", "DS2_Test"])][
        ["Model", "Split", "AUC"]
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.barplot(data=auc_comp, x="Model", y="AUC", hue="Split", ax=ax)
    ax.set_title("AUC Comparison: DS1 Test vs DS2 Test")
    ax.set_ylim(0, 1)
    style_plot(ax)
    st.pyplot(fig)
    st.dataframe(
        auc_comp.pivot(index="Model", columns="Split", values="AUC").reset_index(),
        use_container_width=True,
    )

    st.markdown("**ROC Curves on DS2 Test**")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, (fpr, tpr) in curves_ds2.items():
        auc = metrics_df[
            (metrics_df["Model"] == name) & (metrics_df["Split"] == "DS2_Test")
        ]["AUC"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    style_plot(ax)
    st.pyplot(fig)

    st.markdown("**Drift in Key Vitals (DS1 vs DS2)**")
    vitals = [
        c
        for c in ["mean_bmi", "mean_sbp", "mean_dbp", "mean_heart_rate"]
        if c in ds1.columns and c in ds2.columns
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6))
    for i, col in enumerate(vitals[:4]):
        temp = pd.concat(
            [ds1[[col]].assign(period="DS1"), ds2[[col]].assign(period="DS2")],
            ignore_index=True,
        )
        sns.boxplot(
            data=temp, x="period", y=col, ax=axes[i], palette=["#2563EB", "#D97706"]
        )
        axes[i].set_title(col)
        style_plot(axes[i])
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        """
        <div class="info-box">
        Temporal data drift appears as distributional shifts in vitals and in prevalence patterns.
        This can reduce out-of-time generalization when DS1-trained models are deployed on DS2.
        DS2 now uses chronological holdout (latest timestamps as test) to better simulate real deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_relational_map(data_dir: str):
    st.subheader("Dataset Relationship Map")
    summary_df, edges_df = build_dataset_relational_map(data_dir)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Table Inventory**")
        st.dataframe(summary_df, use_container_width=True)
    with c2:
        st.markdown("**Strongest Join Paths**")
        st.dataframe(edges_df.head(20), use_container_width=True)

    if not edges_df.empty:
        pivot = edges_df.pivot_table(
            index="From", columns="To", values="Overlap_Ratio", aggfunc="max"
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Cross-table key overlap ratio")
        st.pyplot(fig)

    st.markdown(
        """
        <div class="info-box">
        The map uses common healthcare identifiers (PATIENT, ENCOUNTER, PROVIDER, ORGANIZATION, CLAIMID, PAYER, Id)
        and reports overlap ratio to highlight reliable table joins.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_continual_learning(results):
    cl_df = results["cl_df"]
    curves_ds2 = results["curves_ds2"]
    cl_models = results["cl_models"]
    X2_test = results["splits"][5]
    y2_test = results["splits"][7]

    st.markdown(
        """
        <div class="info-box">
        <b>Warm-start MLP:</b> Start from DS1-trained MLP weights, then continue optimization on DS2 train.<br>
        <b>Replay Decision Tree:</b> Retrain a fresh tree on combined DS1 train + DS2 train replay buffer.<br>
        <b>Improvement:</b> DS2-adaptive thresholds are optimized on DS2 train before final DS2 test evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(cl_df, use_container_width=True)

    before_after = pd.DataFrame(
        {
            "Model": ["Decision Tree", "MLP"],
            "Before_CL_AUC": [
                cl_df.loc[
                    cl_df["Model_Variant"] == "Baseline_DT_DS1->DS2", "AUC"
                ].values[0],
                cl_df.loc[
                    cl_df["Model_Variant"] == "Baseline_MLP_DS1->DS2", "AUC"
                ].values[0],
            ],
            "After_CL_AUC": [
                cl_df.loc[cl_df["Model_Variant"] == "Replay_DT_DS1+DS2", "AUC"].values[
                    0
                ],
                cl_df.loc[
                    cl_df["Model_Variant"] == "WarmStart_MLP_DS2_FT", "AUC"
                ].values[0],
            ],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    for i, model in enumerate(["Decision Tree", "MLP"]):
        row = before_after[before_after["Model"] == model].iloc[0]
        sns.barplot(
            x=["Before", "After"],
            y=[row["Before_CL_AUC"], row["After_CL_AUC"]],
            ax=axes[i],
            palette="viridis",
        )
        axes[i].set_ylim(0, 1)
        axes[i].set_title(f"{model} AUC")
        style_plot(axes[i])
    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, (fpr, tpr) in curves_ds2.items():
        ax.plot(fpr, tpr, label=f"Baseline {name}")
    for lbl, mdl in [
        ("Replay DT", cl_models["dt_replay"]),
        ("Warm-start MLP", cl_models["mlp_warmstart"]),
    ]:
        if hasattr(mdl, "predict_proba") and len(np.unique(y2_test)) > 1:
            prob = mdl.predict_proba(X2_test)[:, 1]
            fpr, tpr, _ = roc_curve(y2_test, prob)
            ax.plot(fpr, tpr, linewidth=2.5, label=lbl)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("ROC on DS2 Test: Baseline vs Continual Learning")
    ax.legend()
    style_plot(ax)
    st.pyplot(fig)


def render_feature_interpretation(results):
    dt = results["trained"]["Decision Tree"]
    ds1 = results["ds1"]
    feature_cols = results["feature_cols"]

    feature_names = np.array(feature_cols)
    importances = dt.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    top_n = st.slider(
        "Top N important features",
        min_value=5,
        max_value=min(25, len(imp_df)),
        value=10,
    )
    top_imp = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.barplot(data=top_imp, x="importance", y="feature", ax=ax, palette="Blues_r")
    ax.set_title("Decision Tree Feature Importances")
    style_plot(ax)
    st.pyplot(fig)

    mapping = pd.DataFrame(
        {
            "Feature Pattern": [
                "mean_bmi",
                "mean_sbp",
                "mean_dbp",
                "mean_heart_rate",
                "encounter_count",
                "medication_count",
            ],
            "Clinical Interpretation": [
                "BMI elevation aligns with metabolic syndrome risk",
                "High systolic BP indicates hypertension burden",
                "High diastolic BP supports cardiovascular strain",
                "Elevated resting heart rate can signal cardiorespiratory stress",
                "Higher utilization reflects complex chronic care needs",
                "Medication intensity often tracks chronic disease management",
            ],
        }
    )
    st.dataframe(mapping, use_container_width=True)

    st.markdown("**Decision Tree Visual (max depth 3)**")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    plot_tree(
        dt.named_steps["clf"],
        feature_names=feature_cols,
        class_names=["No Chronic", "Chronic"],
        filled=True,
        max_depth=3,
        fontsize=7,
        ax=ax,
    )
    st.pyplot(fig)

    st.markdown(
        "- **SVM interpretation:** uses weighted margins in transformed feature space; high-impact vitals and utilization features shift support vectors.\n"
        "- **MLP interpretation:** learns nonlinear interactions among vital trends, demographics, and usage intensity."
    )

    st.markdown("**DS1 vs DS2 Distribution for Top 6 Important Features**")
    top6 = top_imp["feature"].head(6).tolist()
    ds2 = results["ds2"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5))
    axes = axes.flatten()
    for i, feat in enumerate(top6):
        if feat not in ds1.columns or feat not in ds2.columns:
            axes[i].axis("off")
            continue
        sns.histplot(
            ds1[feat],
            ax=axes[i],
            color="#2563EB",
            kde=True,
            stat="density",
            alpha=0.35,
            label="DS1",
        )
        sns.histplot(
            ds2[feat],
            ax=axes[i],
            color="#D97706",
            kde=True,
            stat="density",
            alpha=0.35,
            label="DS2",
        )
        axes[i].set_title(feat)
        axes[i].legend()
        style_plot(axes[i])
    for j in range(len(top6), 6):
        axes[j].axis("off")
    plt.tight_layout()
    st.pyplot(fig)


def main():
    st.set_page_config(
        page_title="Assignment-2 — Team 24", page_icon="🏥", layout="wide"
    )
    inject_css()
    st.title("Assignment-2 — Team 24")

    data_dir, target_key, page = sidebar()
    resolved_data_dir = resolve_data_dir(data_dir)
    if not resolved_data_dir.exists():
        st.error(f"Data directory not found: {resolved_data_dir}")
        st.stop()

    try:
        results = train_all_models(str(resolved_data_dir), target_key=target_key)
    except Exception as e:
        st.error(f"Failed to build pipeline: {e}")
        st.stop()

    if page == "🏠 Overview":
        render_overview(results)
    elif page == "🕸 Dataset Relational Map":
        render_dataset_relational_map(str(resolved_data_dir))
    elif page == "📊 EDA — Dataset 1":
        render_eda(results["ds1"], "EDA — Dataset 1 (DS1)")
    elif page == "📊 EDA — Dataset 2":
        render_eda(results["ds2"], "EDA — Dataset 2 (DS2)")
    elif page == "🤖 Model Training & Evaluation":
        render_model_training(results)
    elif page == "📉 Temporal Shift Analysis":
        render_temporal_shift(results)
    elif page == "🔄 Continual Learning":
        render_continual_learning(results)
    elif page == "🌳 Feature Importance & Interpretation":
        render_feature_interpretation(results)

    st.markdown(
        """
        <div class="footer">
        BITS F464 Machine Learning | Assignment 2 | Team 24<br>
        Automated Clinical Prediction under Temporal Shift
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
