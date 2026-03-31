"""
model_utils.py  —  PitWall Analytics Group Assignment
─────────────────────────────────────────────────────
Feature engineering + all ML models:
  • 6 classifiers compared (RF, LR, DT, KNN, SVM, NB)
  • KMeans segmentation
  • Linear / Ridge / Lasso regression
  • Apriori-style association rule mining
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import combinations
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(subs: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    agg = (
        sess.groupby("Subscriber Id")
        .agg(
            total_sessions      = ("Session Duration Min", "count"),
            avg_engagement      = ("Engagement Score",     "mean"),
            std_engagement      = ("Engagement Score",     "std"),
            avg_duration        = ("Session Duration Min", "mean"),
            total_duration      = ("Session Duration Min", "sum"),
            mobile_sessions     = ("Device",         lambda x: (x == "Mobile").sum()),
            weekend_sessions    = ("Is Weekend",     "sum"),
            high_eng_sessions   = ("Engagement Tier",lambda x: (x == "High").sum()),
        )
        .reset_index()
    )
    agg["mobile_pct"]     = (agg["mobile_sessions"]   / agg["total_sessions"]).round(4)
    agg["weekend_pct"]    = (agg["weekend_sessions"]  / agg["total_sessions"]).round(4)
    agg["high_eng_pct"]   = (agg["high_eng_sessions"] / agg["total_sessions"]).round(4)
    agg["std_engagement"] = agg["std_engagement"].fillna(0)

    top_content = (
        sess.groupby("Subscriber Id")["Content Type"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index().rename(columns={"Content Type": "top_content"})
    )

    df = subs.merge(agg, on="Subscriber Id", how="left")
    df = df.merge(top_content, on="Subscriber Id", how="left")

    for c in ["total_sessions","avg_engagement","std_engagement",
              "avg_duration","total_duration","mobile_pct","weekend_pct","high_eng_pct"]:
        df[c] = df[c].fillna(0)
    df["top_content"] = df["top_content"].fillna("Unknown")

    for src, dst in {
        "Plan": "plan_enc", "Region": "region_enc",
        "Acquisition Channel": "channel_enc",
        "Age Group": "age_group_enc", "top_content": "content_enc",
    }.items():
        df[dst] = LabelEncoder().fit_transform(df[src].astype(str))

    return df


FEATURE_COLS = [
    "plan_enc","Monthly Price Usd","region_enc","channel_enc","Age","age_group_enc",
    "Tenure Months","Renewal Count","Nps Score","total_sessions","avg_engagement",
    "std_engagement","avg_duration","mobile_pct","weekend_pct","high_eng_pct","content_enc",
]

FEATURE_LABELS = [
    "Plan Tier","Monthly Price","Region","Acquisition Channel","Age","Age Group",
    "Tenure (months)","Renewal Count","NPS Score","Total Sessions","Avg Engagement Score",
    "Engagement Variability","Avg Session Duration","Mobile Usage %","Weekend Usage %",
    "High-Engagement Session %","Top Content Type",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  ALL CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════════

def train_all_classifiers(df: pd.DataFrame):
    """
    Train 6 classifiers and return comparison metrics + per-model details.
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler   = StandardScaler()
    Xtr_sc   = scaler.fit_transform(X_train)
    Xte_sc   = scaler.transform(X_test)

    classifiers = [
        ("Random Forest",          RandomForestClassifier(n_estimators=300, max_depth=8,
                                    min_samples_leaf=4, class_weight="balanced",
                                    random_state=42, n_jobs=-1),  X_train, X_test),
        ("Logistic Regression",    LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=42),              Xtr_sc,  Xte_sc),
        ("Decision Tree",          DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                    random_state=42),              X_train, X_test),
        ("K-Nearest Neighbours",   KNeighborsClassifier(n_neighbors=7, n_jobs=-1), Xtr_sc, Xte_sc),
        ("Support Vector Machine", SVC(probability=True, class_weight="balanced",
                                    random_state=42, kernel="rbf"), Xtr_sc, Xte_sc),
        ("Naive Bayes",            GaussianNB(),                    Xtr_sc,  Xte_sc),
    ]

    results   = []
    df_scored = None
    imp_df    = None

    for name, clf, Xtr, Xte in classifiers:
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm           = confusion_matrix(y_test, y_pred)

        results.append({
            "name":      name,
            "accuracy":  round(accuracy_score(y_test, y_pred),                  4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0),4),
            "recall":    round(recall_score(y_test,   y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test,       y_pred, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_test,  y_prob),                  4),
            "cm": cm, "fpr": fpr, "tpr": tpr,
            "y_pred": y_pred, "y_prob": y_prob,
        })

        if name == "Random Forest":
            df_scored = df.copy()
            df_scored["churn_prob"] = clf.predict_proba(X.fillna(0))[:, 1]
            df_scored["churn_pred"] = clf.predict(X.fillna(0))
            imp_df = (
                pd.DataFrame({"feature": FEATURE_LABELS,
                              "importance": clf.feature_importances_})
                .sort_values("importance", ascending=True).reset_index(drop=True)
            )

    return results, X_test, y_test, df_scored, imp_df


def get_model_metrics(y_test, y_pred, y_prob) -> dict:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    return {
        "accuracy":   round(accuracy_score(y_test,  y_pred),                   4),
        "precision":  round(precision_score(y_test, y_pred, zero_division=0),  4),
        "recall":     round(recall_score(y_test,    y_pred, zero_division=0),  4),
        "f1":         round(f1_score(y_test,        y_pred, zero_division=0),  4),
        "auc":        round(roc_auc_score(y_test,   y_prob),                   4),
        "cm":         confusion_matrix(y_test, y_pred),
        "fpr": fpr, "tpr": tpr, "prec_curve": prec_c, "rec_curve": rec_c,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  KMEANS SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    feats    = ["avg_engagement","avg_duration","total_sessions",
                "Tenure Months","mobile_pct","high_eng_pct"]
    X_scaled = StandardScaler().fit_transform(df[feats].fillna(0))
    km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df       = df.copy()
    df["segment"] = km.fit_predict(X_scaled)
    rank = (df.groupby("segment")["avg_engagement"].mean()
              .sort_values(ascending=False).index.tolist())
    labels = ["Champions","Engaged","At Risk","Dormant"]
    df["segment_label"] = df["segment"].map({s: l for s, l in zip(rank, labels)})
    return df


def get_elbow_data(df: pd.DataFrame, max_k: int = 10) -> pd.DataFrame:
    feats    = ["avg_engagement","avg_duration","total_sessions",
                "Tenure Months","mobile_pct","high_eng_pct"]
    X_scaled = StandardScaler().fit_transform(df[feats].fillna(0))
    inertias, silhouettes = [], []
    for k in range(2, max_k + 1):
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labs))
    return pd.DataFrame({"k": list(range(2, max_k + 1)),
                         "inertia": inertias, "silhouette": silhouettes})


# ═══════════════════════════════════════════════════════════════════════════════
#  REGRESSION — Linear / Ridge / Lasso
# ═══════════════════════════════════════════════════════════════════════════════

def train_regression_models(df: pd.DataFrame):
    """
    Target: Lifetime Revenue Usd (continuous)
    """
    reg_feats = [
        "Monthly Price Usd","Tenure Months","Renewal Count","Nps Score",
        "total_sessions","avg_engagement","avg_duration","mobile_pct",
        "high_eng_pct","plan_enc","region_enc","channel_enc",
    ]
    feat_labels = [
        "Monthly Price","Tenure Months","Renewal Count","NPS Score",
        "Total Sessions","Avg Engagement","Avg Duration","Mobile %",
        "High Eng %","Plan Tier","Region","Acq Channel",
    ]
    X = df[reg_feats].fillna(0)
    y = df["Lifetime Revenue Usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)

    models = [
        ("Linear Regression", LinearRegression(),              Xtr_sc, Xte_sc),
        ("Ridge Regression",  Ridge(alpha=10.0),               Xtr_sc, Xte_sc),
        ("Lasso Regression",  Lasso(alpha=1.0, max_iter=5000), Xtr_sc, Xte_sc),
    ]

    results = []
    for name, mdl, Xtr, Xte in models:
        mdl.fit(Xtr, y_train)
        y_pred = mdl.predict(Xte)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2     = float(r2_score(y_test, y_pred))
        coefs  = pd.DataFrame({
            "feature":     feat_labels,
            "coefficient": mdl.coef_,
        }).sort_values("coefficient", key=abs, ascending=False)
        results.append({
            "name":   name,
            "r2":     round(r2, 4),
            "rmse":   round(rmse, 2),
            "mae":    round(float(np.mean(np.abs(y_test - y_pred))), 2),
            "y_test": y_test.values,
            "y_pred": y_pred,
            "coefs":  coefs,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ASSOCIATION RULE MINING  (pure Python — no mlxtend needed)
# ═══════════════════════════════════════════════════════════════════════════════

def mine_association_rules(
    sess: pd.DataFrame,
    subs: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.30,
    min_lift: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Subscriber-level basket = unique content types + Plan tier + Churn status.
    Returns (rules_df, item_freq_df).
    """
    sess_s = sess.merge(subs[["Subscriber Id","Plan","Churned"]], on="Subscriber Id", how="left")
    basket = (
        sess_s.groupby("Subscriber Id")["Content Type"]
        .apply(lambda x: list(x.unique()))
        .reset_index(name="items")
    )
    subs_extra = subs[["Subscriber Id","Plan","Churned"]].copy()
    subs_extra["plan_item"]  = "Plan:" + subs_extra["Plan"]
    subs_extra["churn_item"] = "Churn:" + subs_extra["Churned"]
    basket = basket.merge(
        subs_extra[["Subscriber Id","plan_item","churn_item"]],
        on="Subscriber Id", how="left"
    )
    basket["items"] = basket.apply(
        lambda r: r["items"] + [r["plan_item"], r["churn_item"]], axis=1
    )

    transactions = basket["items"].tolist()
    n = len(transactions)

    item_counts: dict[str, int] = defaultdict(int)
    for t in transactions:
        for item in set(t):
            item_counts[item] += 1

    freq_items = {item: cnt / n for item, cnt in item_counts.items()
                  if cnt / n >= min_support}

    pair_counts: dict[tuple, int] = defaultdict(int)
    for t in transactions:
        filtered = sorted([i for i in set(t) if i in freq_items])
        for a, b in combinations(filtered, 2):
            pair_counts[(a, b)] += 1

    rules = []
    for (a, b), cnt in pair_counts.items():
        sup = cnt / n
        if sup < min_support:
            continue
        for ant, con in [(a, b), (b, a)]:
            conf = sup / freq_items[ant]
            if conf < min_confidence:
                continue
            lift = conf / freq_items[con]
            if lift < min_lift:
                continue
            conv = (1 - freq_items[con]) / max(1 - conf, 1e-9)
            rules.append({
                "antecedent":  ant,
                "consequent":  con,
                "support":     round(sup,  4),
                "confidence":  round(conf, 4),
                "lift":        round(lift, 4),
                "conviction":  round(min(conv, 999), 3),
            })

    rules_df = (
        pd.DataFrame(rules)
        .sort_values("lift", ascending=False)
        .drop_duplicates(subset=["antecedent","consequent"])
        .reset_index(drop=True)
        if rules else
        pd.DataFrame(columns=["antecedent","consequent","support","confidence","lift","conviction"])
    )

    item_freq_df = pd.DataFrame([
        {"item": k, "support": round(v, 4)}
        for k, v in sorted(freq_items.items(), key=lambda x: -x[1])
    ])
    return rules_df, item_freq_df


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKWARDS COMPATIBILITY SHIM  (for tab3_predictive + tab4_prescriptive)
# ═══════════════════════════════════════════════════════════════════════════════

def train_churn_model(df: pd.DataFrame):
    """
    Thin wrapper around train_all_classifiers that returns the same
    signature expected by tab3_predictive and tab4_prescriptive.
    Returns:
        clf, X_train, X_test, y_train, y_test, y_pred, y_prob, imp_df, df_scored
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    imp_df = (
        pd.DataFrame({"feature": FEATURE_LABELS, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=True).reset_index(drop=True)
    )

    df_scored = df.copy()
    df_scored["churn_prob"] = clf.predict_proba(X.fillna(0))[:, 1]
    df_scored["churn_pred"] = clf.predict(X.fillna(0))

    return clf, X_train, X_test, y_train, y_test, y_pred, y_prob, imp_df, df_scored
