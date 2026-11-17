"""GCI Home Credit pipeline. See CITATIONS.md for feature and CV references.

Only competition-provided train/test/sample files are read. Audit labels are
reserved until evaluate_audit is explicitly enabled. No competition upload occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import time
import zipfile
import tempfile
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PIPELINE_VERSION = "gci-setup-v1"
EXPECTED_HASHES = {
    "train.csv": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test.csv": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample_submission.csv": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "xgboost", "optuna", "joblib")
GROUPS = {"scores", "affordability", "tenure", "missingness", "peer", "categories", "logs",
          "loan_terms", "peer_profile", "category_crosses", "inquiry_patterns", "pricing_proxies"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=path.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(serialized)
        if temporary.read_text() != serialized:
            raise ValueError("JSON write verification failed")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def promote_submission(source, destination, expected_sha256):
    """Keep the previous complete CSV until its verified replacement is ready."""
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".submission-", suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        if sha256(temporary) != expected_sha256:
            raise ValueError("Submission copy verification failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class Config:
    label: str = "B01_all_features"
    mode: str = "development"
    models: tuple = ("lightgbm", "catboost")
    feature_groups: tuple = ()
    seed: int = 42
    audit_seed: int = 20261009
    audit_fraction: float = 0.20
    n_splits: int = 5
    max_iterations: int = 2000
    early_stopping: int = 150
    threads: int = 4
    evaluate_audit: bool = False
    verify_input_hashes: bool = True
    frozen_choices: bool = False
    model_params: dict = field(default_factory=dict)
    model_options: dict = field(default_factory=dict)
    blend_weights: dict | None = None

    def __post_init__(self):
        self.models = tuple(self.models)
        self.feature_groups = tuple(self.feature_groups)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.label):
            raise ValueError("Use letters, digits, underscores or hyphens in label.")
        if self.mode not in {"development", "final"}:
            raise ValueError("mode must be development or final")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("Select unique models")
        if set(self.models) - {"lightgbm", "catboost", "xgboost"}:
            raise ValueError("Supported models: LightGBM, CatBoost and XGBoost")
        if set(self.feature_groups) - GROUPS:
            raise ValueError("Unknown feature group")
        for name, options in self.model_options.items():
            if name not in self.models or set(options) - {"feature_groups", "max_iterations", "early_stopping", "threads"}:
                raise ValueError("model_options may override only features and training limits for selected models")
            if set(options.get("feature_groups", ())) - GROUPS:
                raise ValueError("Unknown per-model feature group")
            if any(not isinstance(v, int) or v < 1 for k, v in options.items() if k != "feature_groups"):
                raise ValueError("Per-model training limits must be positive integers")
        if not 0 < self.audit_fraction < 0.5 or self.n_splits < 2:
            raise ValueError("Invalid validation split")
        if self.mode == "final" and not self.frozen_choices:
            raise ValueError("Set frozen_choices=True only after model/feature selection is finished")
        if self.mode == "final" and not self.blend_weights:
            raise ValueError("Final mode requires the previously selected blend_weights")
        if self.mode == "final" and self.evaluate_audit:
            raise ValueError("Final training uses former audit rows; an audit score would not be independent")
        if self.evaluate_audit and not self.frozen_choices:
            raise ValueError("Audit evaluation requires frozen choices")
        if self.evaluate_audit and self.blend_weights is None:
            raise ValueError("Audit evaluation requires explicit frozen blend weights")
        if self.blend_weights is not None:
            weights = self.blend_weights
            if set(weights) - set(self.models) or any(v < 0 for v in weights.values()):
                raise ValueError("Blend weights must be nonnegative and refer to trained models")
            if not np.isclose(sum(weights.values()), 1.0):
                raise ValueError("Blend weights must sum to one")


def load_inputs(input_dir, verify_hashes=True):
    input_dir = Path(input_dir)
    paths = {name: input_dir / name for name in EXPECTED_HASHES}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing competition input: {path}")
    hashes = {name: sha256(path) for name, path in paths.items()}
    if verify_hashes and hashes != EXPECTED_HASHES:
        raise ValueError("Input bytes differ from the audited competition files. Check the folder/version before training.")
    train, test, sample = (pd.read_csv(paths[name]) for name in EXPECTED_HASHES)
    if "TARGET" not in train or "TARGET" in test:
        raise ValueError("Unexpected target schema")
    if not train.drop(columns="TARGET").columns.equals(test.columns):
        raise ValueError("Training/test predictor schema mismatch")
    if sample.columns.tolist() != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Sample submission columns must be SK_ID_CURR,TARGET")
    for frame in (train, test, sample):
        if frame.SK_ID_CURR.isna().any() or frame.SK_ID_CURR.duplicated().any():
            raise ValueError("Missing or duplicate IDs")
    if set(train.SK_ID_CURR) & set(test.SK_ID_CURR):
        raise ValueError("Training and test IDs overlap")
    if set(sample.SK_ID_CURR) != set(test.SK_ID_CURR):
        raise ValueError("Sample/test ID sets differ")
    if set(train.TARGET.unique()) != {0, 1}:
        raise ValueError("Expected both binary target classes")
    return train, test, sample, hashes


def safe_ratio(numerator, denominator):
    valid = denominator.notna() & np.isfinite(denominator) & denominator.abs().gt(1e-12)
    return numerator.div(denominator.where(valid)).replace([np.inf, -np.inf], np.nan)


def make_features(frame, groups=()):
    """Row-wise transformations only. References: Koehrsen and Aguiar."""
    raw = frame.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore").copy()
    x = raw.copy()
    x["DAYS_EMPLOYED_ANOM"] = raw.DAYS_EMPLOYED.eq(365243).astype("int8")
    x["DAYS_EMPLOYED"] = raw.DAYS_EMPLOYED.mask(raw.DAYS_EMPLOYED.eq(365243))
    if "scores" in groups:
        scores = x[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        for name, fn in (("MEAN", "mean"), ("MIN", "min"), ("MAX", "max")):
            x[f"EXT_{name}"] = getattr(scores, fn)(axis=1)
        x["EXT_STD"] = scores.std(axis=1, ddof=0)
        x["EXT_COUNT"] = scores.notna().sum(axis=1)
        for a, b in ((1, 2), (1, 3), (2, 3)):
            x[f"EXT_{a}_{b}_PRODUCT"] = scores[f"EXT_SOURCE_{a}"] * scores[f"EXT_SOURCE_{b}"]
            x[f"EXT_{a}_{b}_DIFF"] = scores[f"EXT_SOURCE_{a}"] - scores[f"EXT_SOURCE_{b}"]
    if "affordability" in groups:
        for name, a, b in (
            ("CREDIT_INCOME", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
            ("ANNUITY_INCOME", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
            ("CREDIT_ANNUITY", "AMT_CREDIT", "AMT_ANNUITY"),
            ("CREDIT_GOODS", "AMT_CREDIT", "AMT_GOODS_PRICE"),
            ("INCOME_PER_FAMILY", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
            ("CREDIT_PER_FAMILY", "AMT_CREDIT", "CNT_FAM_MEMBERS"),
            ("ANNUITY_PER_FAMILY", "AMT_ANNUITY", "CNT_FAM_MEMBERS"),
        ):
            x[name] = safe_ratio(x[a], x[b])
        x["CREDIT_GOODS_DIFF"] = x.AMT_CREDIT - x.AMT_GOODS_PRICE
    if "tenure" in groups:
        x["AGE_YEARS"] = -x.DAYS_BIRTH / 365.25
        x["TENURE_YEARS"] = -x.DAYS_EMPLOYED / 365.25
        x["TENURE_AGE_RATIO"] = safe_ratio(x.TENURE_YEARS, x.AGE_YEARS)
        for c in ("DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"):
            x[f"{c}_AGE_RATIO"] = safe_ratio(x[c], x.DAYS_BIRTH)
    if "missingness" in groups:
        x["N_MISSING"] = raw.isna().sum(axis=1)
        for c in raw:
            if c in {"OWN_CAR_AGE", "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"} or c.startswith("AMT_REQ_"):
                x[f"{c}_MISSING"] = raw[c].isna().astype("int8")
        x["CAR_AGE_60_PLUS"] = raw.OWN_CAR_AGE.ge(60).astype("int8")
        x["PHONE_CHANGE_ZERO"] = raw.DAYS_LAST_PHONE_CHANGE.eq(0).astype("int8")
    if "categories" in groups:
        x["CONTRACT_INCOME"] = x.NAME_CONTRACT_TYPE.fillna("__MISSING__") + " / " + x.NAME_INCOME_TYPE.fillna("__MISSING__")
    if "logs" in groups:
        for c in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"):
            x[f"LOG_{c}"] = np.log1p(x[c].where(x[c].ge(0)))
    if "loan_terms" in groups:
        # Current-application mathematical ideas: Onodera; see CITATIONS.md.
        adults = (x.CNT_FAM_MEMBERS - x.CNT_CHILDREN).where(lambda s: s.gt(0))
        family = x.CNT_FAM_MEMBERS.where(x.CNT_FAM_MEMBERS.gt(0))
        x["ADULT_COUNT"] = adults
        x["CHILD_SHARE"] = safe_ratio(x.CNT_CHILDREN, family)
        for name, amount in (("INCOME", "AMT_INCOME_TOTAL"), ("CREDIT", "AMT_CREDIT"),
                             ("ANNUITY", "AMT_ANNUITY"), ("GOODS", "AMT_GOODS_PRICE")):
            x[f"{name}_PER_ADULT"] = safe_ratio(x[amount], adults)
        x["GOODS_INCOME"] = safe_ratio(x.AMT_GOODS_PRICE, x.AMT_INCOME_TOTAL)
        x["GOODS_ANNUITY"] = safe_ratio(x.AMT_GOODS_PRICE, x.AMT_ANNUITY)
        x["FINANCING_INCOME"] = safe_ratio(x.AMT_CREDIT - x.AMT_GOODS_PRICE, x.AMT_INCOME_TOTAL)
        x["GOODS_PER_FAMILY"] = safe_ratio(x.AMT_GOODS_PRICE, family)
        x["PAYMENT_RATE"] = safe_ratio(x.AMT_ANNUITY, x.AMT_CREDIT)
        x["AGE_AT_REPAYMENT_PROXY"] = (-x.DAYS_BIRTH + 30 * safe_ratio(x.AMT_CREDIT, x.AMT_ANNUITY)) / 365.25
    if "peer_profile" in groups:
        x["CREDIT_ANNUITY"] = safe_ratio(x.AMT_CREDIT, x.AMT_ANNUITY)
        x["EXT_PROFILE_MEAN"] = x[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    if "category_crosses" in groups:
        age = pd.cut(-x.DAYS_BIRTH / 365.25, [0, 25, 35, 45, 55, 65, np.inf], include_lowest=True).astype(str)
        term = pd.cut(safe_ratio(x.AMT_CREDIT, x.AMT_ANNUITY),
                      [-np.inf, 6, 12, 18, 24, 36, 60, np.inf]).astype(str)
        for label, columns in (
            ("HOUSING_FAMILY", [x.NAME_HOUSING_TYPE, x.NAME_FAMILY_STATUS]),
            ("EDUCATION_OCCUPATION", [x.NAME_EDUCATION_TYPE, x.OCCUPATION_TYPE]),
            ("OCCUPATION_ORGANIZATION", [x.OCCUPATION_TYPE, x.ORGANIZATION_TYPE]),
            ("INCOME_EDUCATION", [x.NAME_INCOME_TYPE, x.NAME_EDUCATION_TYPE]),
            ("INCOME_CONTRACT", [x.NAME_INCOME_TYPE, x.NAME_CONTRACT_TYPE]),
            ("AGE_FAMILY", [age, x.NAME_FAMILY_STATUS]),
            ("TERM_CONTRACT", [term, x.NAME_CONTRACT_TYPE]),
            ("REGION_HOUSING", [x.REGION_RATING_CLIENT, x.NAME_HOUSING_TYPE]),
        ):
            # JSON tuples avoid collisions between category strings containing separators.
            parts = [s.astype("string").fillna("__MISSING__") for s in columns]
            x[f"CROSS_{label}"] = [json.dumps(v, ensure_ascii=True) for v in zip(*parts)]
    if "inquiry_patterns" in groups:
        hour, month, quarter, year = [x[f"AMT_REQ_CREDIT_BUREAU_{s}"].where(
            x[f"AMT_REQ_CREDIT_BUREAU_{s}"].ge(0)) for s in ("HOUR", "MON", "QRT", "YEAR")]
        # Comparisons of reported windows, not reconstructed monthly observations.
        x["ENQUIRY_MONTH_TO_ANNUAL_RATE"] = safe_ratio(month, year / 12 + 1)
        x["ENQUIRY_QUARTER_TO_ANNUAL_RATE"] = safe_ratio(quarter, year / 4 + 1)
        x["ENQUIRY_MONTH_TO_QUARTER_RATE"] = safe_ratio(month, quarter / 3 + 1)
        x["ENQUIRY_HOUR_TO_MONTH"] = safe_ratio(hour, month + 1)
        x["ENQUIRY_RECENT_ACTIVITY"] = month.gt(0).astype(float).where(month.notna())
        x["ENQUIRY_MISSING_PATTERN"] = ["".join(v) for v in zip(
            *[s.isna().astype(int).astype(str) for s in (hour, month, quarter, year)])]
    if "pricing_proxies" in groups:
        # Godula-inspired scenarios; rate/term are unknown, never asserted as observed.
        payment_ratio = safe_ratio(x.AMT_ANNUITY.where(x.AMT_ANNUITY.gt(0)),
                                   x.AMT_CREDIT.where(x.AMT_CREDIT.gt(0)))
        x["ZERO_RATE_TERM_PROXY"] = safe_ratio(pd.Series(1., index=x.index), payment_ratio)
        scenario_columns = []
        for months in (12, 24, 36, 60):
            low, high = np.zeros(len(x)), np.full(len(x), .06)
            q = payment_ratio.to_numpy()
            maximum_payment = .06 / (1 - 1.06 ** -months)
            valid = np.isfinite(q) & (q >= 1 / months) & (q <= maximum_payment)
            for _ in range(32):
                rate = (low + high) / 2
                implied_payment = rate / (-np.expm1(-months * np.log1p(rate)))
                low = np.where(implied_payment < q, rate, low)
                high = np.where(implied_payment >= q, rate, high)
            name = f"RATE_HYPOTHESIS_{months}M"
            x[name] = np.where(valid, 12 * (low + high) / 2, np.nan)
            scenario_columns.append(name)
        x["FEASIBLE_TERM_SCENARIOS"] = x[scenario_columns].notna().sum(axis=1).astype(float).where(payment_ratio.notna())
    return x.replace([np.inf, -np.inf], np.nan)


class FeatureProcessor:
    """Fold-trained categories and optional peer/frequency statistics."""
    def __init__(self, groups, model_name):
        self.groups, self.model_name = tuple(groups), model_name

    def fit(self, x):
        self.categorical = x.select_dtypes(include=["object", "string"]).columns.tolist()
        self.categories = {c: sorted(x[c].dropna().astype(str).unique().tolist()) for c in self.categorical}
        self.peers = {}
        self.frequencies = {}
        if "peer" in self.groups:
            for c in ("OCCUPATION_TYPE", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE"):
                key = x[c].fillna("__MISSING__")
                summary = x.groupby(key, observed=True).AMT_INCOME_TOTAL.agg(["median", "count"])
                self.peers[c] = summary.loc[summary["count"].ge(20), "median"].to_dict()
            self.income_median = float(x.AMT_INCOME_TOTAL.median())
        if "categories" in self.groups:
            self.frequencies = {c: x[c].fillna("__MISSING__").value_counts().div(len(x)).to_dict() for c in self.categorical}
        if "category_crosses" in self.groups:
            self.frequencies.update({c: x[c].fillna("__MISSING__").value_counts().div(len(x)).to_dict()
                                     for c in self.categorical if c.startswith("CROSS_")})
        self.profiles = []
        if "peer_profile" in self.groups:
            values = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_ANNUITY", "EXT_PROFILE_MEAN"]
            for keys in self.profile_keys(x):
                grouped = x[values].groupby([keys[c] for c in keys], observed=True, dropna=False)
                counts = grouped.size()
                stats = grouped.agg(["median", "std"])
                # Sparse groups fall back to training-wide statistics, never validation/test statistics.
                stats.loc[counts.lt(20), :] = np.nan
                fallback = {(value, stat): float(getattr(x[value], stat)()) for value in values for stat in ("median", "std")}
                self.profiles.append((stats, counts, fallback))
        return self

    @staticmethod
    def profile_keys(x):
        age = pd.cut(-x.DAYS_BIRTH / 365.25, [0, 27, 40, 50, 65, np.inf], include_lowest=True).astype(str)
        term = pd.cut(x.CREDIT_ANNUITY, [-np.inf, 6, 12, 18, 24, 30, 36, np.inf]).astype(str)
        return [pd.DataFrame({"education": x.NAME_EDUCATION_TYPE.fillna("__MISSING__"),
                              "occupation": x.OCCUPATION_TYPE.fillna("__MISSING__"), "age": age}),
                pd.DataFrame({"term": term, "income_type": x.NAME_INCOME_TYPE.fillna("__MISSING__"),
                              "education": x.NAME_EDUCATION_TYPE.fillna("__MISSING__")})]

    def transform(self, x):
        z = x.copy()
        for c, mapping in self.peers.items():
            peer = z[c].fillna("__MISSING__").map(mapping).fillna(self.income_median)
            z[f"{c}_PEER_INCOME"] = peer
            z[f"{c}_RELATIVE_INCOME"] = safe_ratio(z.AMT_INCOME_TOTAL, peer)
        for c, mapping in self.frequencies.items():
            z[f"{c}_FREQUENCY"] = z[c].fillna("__MISSING__").map(mapping).fillna(0)
        if getattr(self, "profiles", []):
            for number, (keys, (stats, counts, fallback)) in enumerate(zip(self.profile_keys(x), self.profiles)):
                index = pd.MultiIndex.from_frame(keys)
                matched = stats.reindex(index)
                z[f"PROFILE_{number}_COUNT"] = counts.reindex(index).fillna(0).to_numpy()
                for (value, stat), default in fallback.items():
                    feature = f"PROFILE_{number}_{value}_{stat.upper()}"
                    z[feature] = matched[(value, stat)].fillna(default).to_numpy()
                    if stat == "median":
                        z[f"PROFILE_{number}_{value}_RELATIVE"] = safe_ratio(z[value], z[feature])
        for c in self.categorical:
            if self.model_name in {"lightgbm", "xgboost"}:
                z[c] = pd.Categorical(z[c], categories=self.categories[c])
            else:
                z[c] = z[c].fillna("__MISSING__").astype(str)
        return z


def model_defaults(model_name, cfg):
    if model_name == "lightgbm":
        return dict(objective="binary", metric="auc", n_estimators=cfg.max_iterations,
                    learning_rate=0.035, num_leaves=31, max_depth=-1, min_child_samples=150,
                    colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
                    reg_alpha=0.05, reg_lambda=5.0, random_state=cfg.seed,
                    deterministic=True, force_col_wise=True, n_jobs=cfg.threads, verbosity=-1)
    if model_name == "xgboost":
        return dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cpu",
                    enable_categorical=True, n_estimators=cfg.max_iterations, learning_rate=.035,
                    max_depth=4, min_child_weight=20, subsample=.85, colsample_bytree=.8,
                    reg_lambda=10., reg_alpha=.1, max_bin=256, n_jobs=cfg.threads,
                    random_state=cfg.seed, early_stopping_rounds=cfg.early_stopping, verbosity=0)
    return dict(loss_function="Logloss", eval_metric="AUC", iterations=cfg.max_iterations,
                learning_rate=0.04, depth=6, l2_leaf_reg=6, random_seed=cfg.seed,
                thread_count=cfg.threads, allow_writing_files=False, verbose=False)


def preflight_backends(cfg):
    """Reject unavailable requested accelerators before fitting any full fold."""
    probe_x = np.asarray([[float(i), float(i % 3)] for i in range(32)], dtype=np.float32)
    probe_y = np.arange(32) % 2
    for name in cfg.models:
        model_cfg = replace(cfg, **cfg.model_options.get(name, {}), model_options={})
        params = {**model_defaults(name, model_cfg), **cfg.model_params.get(name, {})}
        requested = str(params.get("task_type", params.get("device_type", params.get("device", "cpu")))).lower()
        if requested == "cpu":
            continue
        try:
            if name == "xgboost":
                from xgboost import XGBClassifier
                model = XGBClassifier(n_estimators=1, max_depth=1, n_jobs=1, verbosity=0,
                                      tree_method=params["tree_method"], device=params["device"])
                model.fit(probe_x, probe_y)
                actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
                if not str(actual).startswith("cuda"):
                    raise RuntimeError(f"native probe fitted {actual}")
            elif name == "catboost":
                from catboost import CatBoostClassifier
                selectors = {key: params[key] for key in ("task_type", "devices") if key in params}
                model = CatBoostClassifier(iterations=1, depth=2, thread_count=1, verbose=False,
                                           allow_writing_files=False, gpu_ram_part=.1, **selectors)
                model.fit(probe_x, probe_y)
                actual = model.get_all_params()["task_type"]
                if str(actual).lower() != requested:
                    raise RuntimeError(f"native probe fitted {actual}")
            else:
                import lightgbm as lgb
                selectors = {key: params[key] for key in ("device_type", "device", "gpu_device_id",
                             "gpu_platform_id", "num_gpu", "gpu_use_dp") if key in params}
                model = lgb.LGBMClassifier(n_estimators=1, num_leaves=3, max_bin=15,
                                          min_child_samples=1, n_jobs=1, verbosity=-1, **selectors)
                model.fit(probe_x, probe_y)
                actual = model.booster_.params.get("device_type", model.booster_.params.get("device", "cpu"))
                if str(actual).lower() != requested:
                    raise RuntimeError(f"native probe fitted {actual}")
        except Exception as error:
            raise RuntimeError(f"Requested {name} {requested} is unavailable; no full folds started and no CPU fallback") from error


def fit_cv(x, y, test_x, ids, cfg, output_dir=None, audit_x=None, trial=None):
    preflight_backends(cfg)
    result = {"oof": {}, "test": {}, "audit": {}, "metrics": {}}
    folds = list(StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed).split(x, y))
    fold_ids = np.full(len(x), -1, dtype=int)
    for fold, (_, va) in enumerate(folds):
        fold_ids[va] = fold
    result["fold_ids"] = fold_ids
    for name in cfg.models:
        model_cfg = replace(cfg, **cfg.model_options.get(name, {}), model_options={})
        # With per-model options, callers pass raw predictors; each member derives only its selected features.
        model_x = make_features(x, model_cfg.feature_groups) if cfg.model_options else x
        model_test = make_features(test_x, model_cfg.feature_groups) if cfg.model_options and test_x is not None else test_x
        model_audit = make_features(audit_x, model_cfg.feature_groups) if cfg.model_options and audit_x is not None else audit_x
        oof = np.full(len(x), np.nan)
        test_pred = np.zeros(len(test_x)) if test_x is not None else None
        audit_pred = np.zeros(len(audit_x)) if audit_x is not None else None
        fold_metrics = []
        for fold, (tr, va) in enumerate(folds):
            started = time.monotonic()
            processor = FeatureProcessor(model_cfg.feature_groups, name).fit(model_x.iloc[tr])
            train_x, valid_x = processor.transform(model_x.iloc[tr]), processor.transform(model_x.iloc[va])
            feature_count = train_x.shape[1]
            params = {**model_defaults(name, model_cfg), **cfg.model_params.get(name, {})}
            requested_backend = params.get("task_type", params.get("device_type", params.get("device", "CPU")))
            print(f"{name} fold {fold + 1}/{cfg.n_splits}: starting on {requested_backend}", flush=True)
            if name == "lightgbm":
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**params)
                model.fit(train_x, y.iloc[tr], eval_set=[(valid_x, y.iloc[va])],
                          callbacks=[lgb.early_stopping(model_cfg.early_stopping, first_metric_only=True, verbose=False)])
                best_iteration = int(model.best_iteration_ or model_cfg.max_iterations)
                backend = str(model.booster_.params.get("device_type", model.booster_.params.get("device", "CPU"))).upper()
                if str(requested_backend).upper() != backend:
                    raise RuntimeError(f"Requested {requested_backend}, fitted {backend}; no automatic CPU fallback")
            elif name == "xgboost":
                from xgboost import XGBClassifier
                model = XGBClassifier(**params)
                model.fit(train_x, y.iloc[tr], eval_set=[(valid_x, y.iloc[va])], verbose=False)
                device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
                if str(requested_backend).startswith("cuda") and not device.startswith("cuda"):
                    raise RuntimeError(f"Requested {requested_backend}, fitted {device}; no automatic CPU fallback")
                backend = "GPU" if device.startswith("cuda") else "CPU"
                best_iteration = int(model.best_iteration + 1)
            else:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(**params)
                model.fit(train_x, y.iloc[tr], cat_features=processor.categorical,
                          eval_set=(valid_x, y.iloc[va]), early_stopping_rounds=model_cfg.early_stopping, use_best_model=True)
                best_iteration = int(model.tree_count_)
                backend = model.get_all_params()["task_type"]
                if str(requested_backend).upper() != backend:
                    raise RuntimeError(f"Requested {requested_backend}, fitted {backend}")
            oof[va] = model.predict_proba(valid_x)[:, 1]
            if test_pred is not None:
                test_pred += model.predict_proba(processor.transform(model_test))[:, 1] / cfg.n_splits
            if audit_pred is not None:
                audit_pred += model.predict_proba(processor.transform(model_audit))[:, 1] / cfg.n_splits
            metric = {"fold": fold, "auc": float(roc_auc_score(y.iloc[va], oof[va])),
                      "best_iteration": best_iteration, "seconds": round(time.monotonic() - started, 2),
                      "backend": backend}
            fold_metrics.append(metric)
            print(f"{name} fold {fold + 1}/{cfg.n_splits}: AUC={metric['auc']:.6f}, trees={best_iteration}, "
                  f"backend={backend}, seconds={metric['seconds']}", flush=True)
            if output_dir is not None:
                joblib.dump({"processor": processor, "model": model}, Path(output_dir) / f"{name}_fold_{fold}.joblib")
                if name == "xgboost":
                    model.save_model(Path(output_dir) / f"{name}_fold_{fold}.ubj")
                if hasattr(model, "feature_importances_"):
                    pd.DataFrame({"feature": train_x.columns, "importance": model.feature_importances_}).to_csv(Path(output_dir) / f"{name}_importance_{fold}.csv", index=False)
            if trial is not None:
                trial.report(float(np.mean([m["auc"] for m in fold_metrics])), step=fold)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            del model, train_x, valid_x
        if not np.isfinite(oof).all():
            raise ValueError("Incomplete OOF predictions")
        result["oof"][name], result["test"][name], result["audit"][name] = oof, test_pred, audit_pred
        result["metrics"][name] = {"oof_auc": float(roc_auc_score(y, oof)), "folds": fold_metrics,
                                   "feature_count": feature_count}
    return result


def development_split(train, cfg):
    development, audit = train_test_split(np.arange(len(train)), test_size=cfg.audit_fraction,
                                         stratify=train.TARGET, random_state=cfg.audit_seed)
    return np.sort(development), np.sort(audit)


def validate_submission(submission, sample):
    if submission.columns.tolist() != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Submission columns or order changed")
    if not submission.SK_ID_CURR.equals(sample.SK_ID_CURR):
        raise ValueError("Submission IDs are missing, reordered or duplicated")
    probabilities = submission.TARGET.to_numpy()
    if not np.isfinite(probabilities).all() or not submission.TARGET.between(0, 1).all():
        raise ValueError("Every row needs a finite model probability in [0, 1]")
    return True


def make_submission(sample, test_ids, predictions):
    if len(test_ids) != len(predictions) or pd.Series(test_ids).duplicated().any():
        raise ValueError("Prediction IDs and row count mismatch")
    aligned = pd.Series(predictions, index=np.asarray(test_ids)).reindex(sample.SK_ID_CURR)
    sample_sub = sample.copy()
    # Same submission contract as the tutorial: update TARGET and save index=False.
    sample_sub["TARGET"] = aligned.to_numpy()
    validate_submission(sample_sub, sample)
    return sample_sub


def write_report(manifest, note_dir):
    note_dir = Path(note_dir)
    note_dir.mkdir(parents=True, exist_ok=True)
    run_id = manifest["run_id"]
    cfg = manifest["config"]
    score_label = "Development OOF AUC" if cfg["mode"] == "development" else "Final-training OOF AUC (not independent audit)"
    rows = "\n".join(f"| {name} | {values['oof_auc']:.6f} |" for name, values in manifest["model_metrics"].items())
    audit_text = (f"Audit AUC: {manifest['audit_auc']:.6f}. Audit has now been inspected."
                  if manifest["audit_auc"] is not None else "Audit labels were not evaluated.")
    note = f'''---
type: experiment_run
status: completed
run_id: {run_id}
stage: {cfg['mode']}
oof_auc: {manifest['selected_oof_auc']:.8f}
public_auc:
created: {manifest['created_utc'][:10]}
tags: [gci, experiment, generated]
---
# {run_id}

[[00 Dashboard]] · [[03 Experiments]] · [[04 Submission Tracker]]

{manifest['selected_name']} generated {manifest['test_rows']:,} model probabilities. No competition upload occurred.

| Model | {score_label} |
| --- | --- |
{rows}

Selected OOF AUC: **{manifest['selected_oof_auc']:.6f}**. This is a local model-selection result, not a leaderboard score. {audit_text}

- Feature groups: {', '.join(cfg['feature_groups']) or 'raw predictors + employment placeholder flag'}
- Per-model feature/training overrides: `{json.dumps(cfg.get('model_options', {}), sort_keys=True)}`
- Folds / seed: {cfg['n_splits']} / {cfg['seed']}
- Weights: `{json.dumps(manifest['selected_weights'], sort_keys=True)}`
- Runtime: {manifest['seconds']:.1f} seconds
- CSV SHA-256: `{manifest['submission_sha256']}`
- Run artifacts: `{manifest['run_dir']}`

## Decision

- [ ] Compare with the current champion on the same folds.
- [ ] Record keep/reject and the reason in [[06 Decisions]].
- [ ] If submitted, record the public score and upload timestamp in [[04 Submission Tracker]].

## Notes

Write observations here. Generated metrics above are immutable evidence for this run.
'''
    (note_dir / f"{run_id}.md").write_text(note)
    (note_dir / "Latest Run.md").write_text(f"<!-- Generated by the notebook. -->\n# Latest completed run\n\n[[{run_id}|Open run evidence]]\n\nLocal OOF AUC: **{manifest['selected_oof_auc']:.6f}**\n\nModel: {manifest['selected_name']}\n\nPublic score: not recorded. No competition upload occurred.\n")
    return note_dir / f"{run_id}.md"


def run_experiment(input_dir, output_dir, cfg=None, notes_dir=None):
    cfg = cfg or Config()
    started = time.monotonic()
    train, test, sample, hashes = load_inputs(input_dir, cfg.verify_input_hashes)
    development, audit = development_split(train, cfg)
    selected_rows = development if cfg.mode == "development" else np.arange(len(train))
    def features(frame):
        return frame.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore").copy() if cfg.model_options else make_features(frame, cfg.feature_groups)
    x = features(train.iloc[selected_rows].reset_index(drop=True))
    y = train.TARGET.iloc[selected_rows].reset_index(drop=True)
    ids = train.SK_ID_CURR.iloc[selected_rows].reset_index(drop=True)
    test_x = features(test)
    audit_x = features(train.iloc[audit].reset_index(drop=True)) if cfg.evaluate_audit else None
    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%S%fZ") + "_" + cfg.label
    output_dir = Path(output_dir)
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", asdict(cfg))
    executed_source_hash = sha256(__file__)
    shutil.copy2(__file__, run_dir / "gci_pipeline.py")
    results = fit_cv(x, y, test_x, ids, cfg, run_dir, audit_x)
    if cfg.blend_weights is None:
        selected_name = max(results["metrics"], key=lambda name: results["metrics"][name]["oof_auc"])
        weights = {name: float(name == selected_name) for name in cfg.models}
    else:
        weights = {name: cfg.blend_weights.get(name, 0.0) for name in cfg.models}
        selected_name = "probability_blend" if sum(w > 0 for w in weights.values()) > 1 else max(weights, key=weights.get)
    selected_oof = sum(weights[name] * results["oof"][name] for name in cfg.models)
    selected_test = sum(weights[name] * results["test"][name] for name in cfg.models)
    oof = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": y, "fold": results["fold_ids"], **results["oof"], "selected": selected_oof})
    oof.to_csv(run_dir / "oof.csv", index=False)
    assignment = pd.DataFrame({"SK_ID_CURR": train.SK_ID_CURR, "partition": "audit", "fold": -1})
    assignment.loc[selected_rows, "partition"] = "development" if cfg.mode == "development" else "final_training"
    assignment.loc[selected_rows, "fold"] = results["fold_ids"]
    assignment.to_csv(run_dir / "fold_assignments.csv", index=False)
    for name, prediction in results["test"].items():
        make_submission(sample, test.SK_ID_CURR, prediction).to_csv(run_dir / f"submission_{name}.csv", index=False, lineterminator="\n")
    submission = make_submission(sample, test.SK_ID_CURR, selected_test)
    submission.to_csv(run_dir / "submission.csv", index=False, lineterminator="\n")
    validate_submission(pd.read_csv(run_dir / "submission.csv"), sample)
    audit_auc = None
    if cfg.evaluate_audit:
        audit_prediction = sum(weights[name] * results["audit"][name] for name in cfg.models)
        audit_auc = float(roc_auc_score(train.TARGET.iloc[audit], audit_prediction))
        pd.DataFrame({"SK_ID_CURR": train.SK_ID_CURR.iloc[audit], "prediction": audit_prediction}).to_csv(run_dir / "audit_predictions.csv", index=False)
    cfg_frozen = asdict(cfg)
    cfg_frozen["blend_weights"] = weights
    save_json(run_dir / "reproduce_config.json", cfg_frozen)
    package_versions = {name: version(name) for name in PACKAGES}
    if sha256(__file__) != executed_source_hash:
        raise RuntimeError("Pipeline changed during training. Incomplete run was not promoted.")
    (run_dir / "requirements.lock.txt").write_text("\n".join(f"{name}=={value}" for name, value in package_versions.items()) + "\n")
    artifact_hashes = {p.name: sha256(p) for p in run_dir.iterdir() if p.is_file()}
    manifest = {"pipeline_version": PIPELINE_VERSION, "pipeline_sha256": executed_source_hash,
                "run_id": run_id, "created_utc": created.isoformat(), "config": asdict(cfg),
                "input_hashes": hashes, "package_versions": package_versions,
                "training_rows": len(train), "development_rows": len(development), "audit_rows": len(audit),
                "fitted_population_rows": len(selected_rows), "test_rows": len(test),
                "feature_count": max(v.get("feature_count", x.shape[1]) for v in results["metrics"].values()),
                "model_metrics": results["metrics"], "selected_weights": weights, "selected_name": selected_name,
                "selected_oof_auc": float(roc_auc_score(y, selected_oof)), "audit_auc": audit_auc,
                "submission_sha256": sha256(run_dir / "submission.csv"), "competition_upload": False,
                "seconds": round(time.monotonic() - started, 2), "run_dir": str(run_dir.resolve()),
                "artifact_hashes": artifact_hashes}
    save_json(run_dir / "manifest.json", manifest)
    promote_submission(run_dir / "submission.csv", output_dir / "submission.csv", manifest["submission_sha256"])
    save_json(output_dir / "latest_run.json", manifest)
    if notes_dir is not None:
        report = write_report(manifest, notes_dir)
        shutil.copy2(report, run_dir / "report.md")
    ledger_path = output_dir / "experiments.csv"
    ledger_row = {"run_id": run_id, "label": cfg.label, "mode": cfg.mode,
                  "feature_groups": ",".join(cfg.feature_groups), "seed": cfg.seed,
                  "n_splits": cfg.n_splits, "oof_auc": manifest["selected_oof_auc"],
                  "selected_model": selected_name, "public_auc": np.nan,
                  "submission_sha256": manifest["submission_sha256"], "competition_upload": False}
    ledger = pd.concat([pd.read_csv(ledger_path), pd.DataFrame([ledger_row])], ignore_index=True) if ledger_path.exists() else pd.DataFrame([ledger_row])
    ledger.to_csv(ledger_path, index=False)
    print(f"Selected {selected_name}, local OOF AUC={manifest['selected_oof_auc']:.6f}", flush=True)
    print(f"Submission checked: {output_dir / 'submission.csv'}", flush=True)
    return manifest


def verify_saved_predictions(input_dir, run_dir):
    """Recreate a completed run from its trusted local models, without retraining."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = Config(**manifest["config"])
    if manifest["pipeline_sha256"] != sha256(__file__):
        raise ValueError("Use the executed pipeline saved with this run")
    for name, expected in manifest["artifact_hashes"].items():
        if sha256(run_dir / name) != expected:
            raise ValueError(f"Run artifact changed: {name}")
    _, test, sample, hashes = load_inputs(input_dir, cfg.verify_input_hashes)
    if hashes != manifest["input_hashes"]:
        raise ValueError("Run input bytes changed")
    prediction = np.zeros(len(test))
    for name in cfg.models:
        groups = cfg.model_options.get(name, {}).get("feature_groups", cfg.feature_groups)
        x = make_features(test, groups)
        model_prediction = np.zeros(len(test))
        for fold in range(cfg.n_splits):
            saved = joblib.load(run_dir / f"{name}_fold_{fold}.joblib")
            model_prediction += saved["model"].predict_proba(saved["processor"].transform(x))[:, 1] / cfg.n_splits
        prediction += manifest["selected_weights"].get(name, 0) * model_prediction
    reproduced = make_submission(sample, test.SK_ID_CURR, prediction)
    original = pd.read_csv(run_dir / "submission.csv", float_precision="round_trip")
    validate_submission(original, sample)
    error = float(np.max(np.abs(reproduced.TARGET.to_numpy() - original.TARGET.to_numpy())))
    if error > 1e-12:
        raise ValueError(f"Saved models do not reproduce the frozen submission: max error {error}")
    regenerated_hash = hashlib.sha256(reproduced.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    report = {"rows": len(original), "max_absolute_error": error,
              "submission_sha256": sha256(run_dir / "submission.csv"),
              "regenerated_sha256": regenerated_hash,
              "byte_identical": regenerated_hash == manifest["submission_sha256"]}
    save_json(run_dir / "saved_model_verification.json", report)
    return report


def tune_model(input_dir, study_dir, cfg, model_name="lightgbm", n_trials=20, timeout=3600):
    """Persistent seeded Optuna search on development rows only."""
    import optuna
    if cfg.mode != "development" or cfg.evaluate_audit:
        raise ValueError("Tuning is restricted to development data; disable audit evaluation")
    train, _, _, hashes = load_inputs(input_dir, cfg.verify_input_hashes)
    dev, _ = development_split(train, cfg)
    raw = train.iloc[dev].reset_index(drop=True)
    x = raw.drop(columns=["SK_ID_CURR", "TARGET"]) if cfg.model_options else make_features(raw, cfg.feature_groups)
    y = train.TARGET.iloc[dev].reset_index(drop=True)
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    identity = {"pipeline_sha256": sha256(__file__), "input_hashes": hashes,
                "groups": cfg.feature_groups, "model": model_name, "n_splits": 3, "threads": cfg.threads,
                "seed": cfg.seed, "audit_seed": cfg.audit_seed, "audit_fraction": cfg.audit_fraction,
                "max_iterations": cfg.max_iterations, "early_stopping": cfg.early_stopping,
                "model_params": cfg.model_params, "model_options": cfg.model_options,
                "versions": {p: version(p) for p in PACKAGES}}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    study_name = f"{model_name}_{identity_hash}"
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{(study_dir / 'studies.db').resolve()}",
                                direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    save_json(study_dir / f"{study_name}_identity.json", identity)

    def objective(trial):
        if model_name == "lightgbm":
            depth = trial.suggest_int("max_depth", 5, 9)
            params = dict(max_depth=depth, num_leaves=trial.suggest_int("num_leaves", 15, min(63, 2 ** depth)),
                          learning_rate=trial.suggest_float("learning_rate", 0.01, 0.07, log=True),
                          min_child_samples=trial.suggest_int("min_child_samples", 50, 500),
                          colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
                          subsample=trial.suggest_float("subsample", 0.7, 1.0), subsample_freq=1,
                          reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 20, log=True),
                          reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 20, log=True))
        elif model_name == "catboost":
            params = dict(depth=trial.suggest_int("depth", 4, 8),
                          learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                          l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True))
        elif model_name == "xgboost":
            params = dict(max_depth=trial.suggest_int("max_depth", 3, 6),
                          min_child_weight=trial.suggest_float("min_child_weight", 5, 60, log=True),
                          learning_rate=trial.suggest_float("learning_rate", .02, .07, log=True),
                          subsample=trial.suggest_float("subsample", .7, 1.),
                          colsample_bytree=trial.suggest_float("colsample_bytree", .65, 1.),
                          reg_lambda=trial.suggest_float("reg_lambda", 1, 40, log=True),
                          reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 5, log=True))
        else:
            raise ValueError("Unsupported tuning model")
        trial_params = {**cfg.model_params.get(model_name, {}), **params}
        trial_cfg = Config(**{**asdict(cfg), "models": (model_name,), "n_splits": 3,
                              "model_params": {model_name: trial_params}, "blend_weights": None,
                              "model_options": ({model_name: cfg.model_options.get(model_name, {})} if cfg.model_options else {})})
        fitted = fit_cv(x, y, None, train.SK_ID_CURR.iloc[dev], trial_cfg, trial=trial)
        return fitted["metrics"][model_name]["oof_auc"]

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
    best = {**cfg.model_params.get(model_name, {}), **study.best_params}
    if model_name == "lightgbm":
        best["subsample_freq"] = 1
    save_json(study_dir / f"{study_name}_best.json", {"params": best, "selection_auc": study.best_value, "identity": identity})
    study.trials_dataframe().to_csv(study_dir / f"{study_name}_trials.csv", index=False)
    return best


def select_blend(run_dir, seed=42):
    """OOF meta-holdout diagnostic. Independent audit confirmation still needed."""
    run_dir = Path(run_dir)
    oof = pd.read_csv(run_dir / "oof.csv")
    models = [name for name in ("lightgbm", "catboost", "xgboost") if name in oof]
    if len(models) < 2:
        raise ValueError("At least two model OOF columns are required")
    fit, validation = train_test_split(np.arange(len(oof)), stratify=oof.TARGET, test_size=0.30, random_state=seed + 17)
    rows = []
    units = 10 if len(models) == 2 else 4
    for integers in product(range(units + 1), repeat=len(models)):
        if sum(integers) != units:
            continue
        candidate = dict(zip(models, (value / units for value in integers)))
        p = sum(candidate[name] * oof[name] for name in models)
        rows.append({**{name + "_weight": float(value) for name, value in candidate.items()},
                     "selection_auc": float(roc_auc_score(oof.TARGET.iloc[fit], p.iloc[fit])),
                     "meta_validation_auc": float(roc_auc_score(oof.TARGET.iloc[validation], p.iloc[validation]))})
    table = pd.DataFrame(rows)
    best = table.sort_values("selection_auc", ascending=False).iloc[0]
    table.to_csv(run_dir / "blend_candidates.csv", index=False)
    weights = {name: float(best[name + "_weight"]) for name in models}
    save_json(run_dir / "blend_proposal.json", {"weights": weights, "selected_by": "selection_auc",
             "meta_validation_auc": float(best.meta_validation_auc),
             "limitation": "Base OOF models share training labels across meta partitions. Confirm on the reserved audit set."})
    return weights, table


def build_bundle(project_dir, manifest, notebook_path=None):
    """Package executed pipeline and frozen run config, not just an unsaved notebook."""
    project_dir = Path(project_dir)
    run_dir = Path(manifest["run_dir"])
    if sha256(__file__) != manifest["pipeline_sha256"]:
        raise ValueError("Pipeline changed since the run. Package the executed version or rerun.")
    if sha256(run_dir / "submission.csv") != manifest["submission_sha256"]:
        raise ValueError("Run submission bytes changed")
    for name in ("reproduce_config.json", "requirements.lock.txt", "config.json"):
        expected = manifest.get("artifact_hashes", {}).get(name)
        if expected is not None and sha256(run_dir / name) != expected:
            raise ValueError(f"Run artifact changed: {name}")
    notebook = Path(notebook_path) if notebook_path is not None else project_dir / "comp.ipynb"
    if not notebook.is_file():
        raise FileNotFoundError("Save comp.ipynb in the Competition folder before creating the code ZIP")
    nb = json.loads(notebook.read_text())
    embedded_sources = []
    for cell in nb.get("cells", []):
        source = cell.get("source", [])
        source = source if isinstance(source, str) else "".join(source)
        lines = source.splitlines(keepends=True)
        if lines and lines[0].strip() == "%%writefile gci_pipeline.py":
            embedded_sources.append("".join(lines[1:]))
    if len(embedded_sources) != 1 or hashlib.sha256(embedded_sources[0].encode()).hexdigest() != manifest["pipeline_sha256"]:
        raise ValueError("Saved notebook embeds a different pipeline. Save the current comp.ipynb before packaging.")
    bundle_path = project_dir / "comp.zip"
    citations_path = project_dir / "CITATIONS.md"
    if not citations_path.is_file():
        raise FileNotFoundError("CITATIONS.md must accompany the code ZIP")
    items = {"comp.ipynb": notebook, "gci_pipeline.py": Path(__file__),
             "reproduce_config.json": run_dir / "reproduce_config.json", "manifest.json": run_dir / "manifest.json",
             "requirements.txt": run_dir / "requirements.lock.txt", "submission.csv": run_dir / "submission.csv"}
    readme = """# Reproduce this GCI submission\n\nPython 3.12 is recommended. Install requirements.txt.\nPut the three competition-provided CSVs in input/. Then run:\n\npython gci_pipeline.py --input input --output reproduced --config reproduce_config.json\n\nThis command uses the executed pipeline and frozen configuration. Compare reproduced/submission.csv with the bundled CSV. Byte identity can depend on platform/library versions. The manifest records the original hash. comp.ipynb offers the Colab interface. No external data or manual overrides are used. No Omnicampus upload is automated.\n"""
    with tempfile.NamedTemporaryFile(dir=project_dir, prefix=".comp-", suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, path in items.items():
                archive.write(path, arcname=name)
            archive.writestr("README.md", readme)
            archive.write(citations_path, "CITATIONS.md")
            archive.writestr("bundle_hashes.json", json.dumps({name: sha256(path) for name, path in items.items()}, indent=2))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or hashlib.sha256(archive.read("submission.csv")).hexdigest() != manifest["submission_sha256"]:
                raise ValueError("ZIP integrity check failed")
        shutil.copy2(temporary, run_dir / "comp.zip")
        temporary.replace(bundle_path)
    finally:
        temporary.unlink(missing_ok=True)
    return bundle_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output")
    parser.add_argument("--config")
    parser.add_argument("--notes")
    args = parser.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
    run_experiment(args.input, args.output, cfg, args.notes)


if __name__ == "__main__":
    main()
