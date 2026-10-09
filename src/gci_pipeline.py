"""GCI Home Credit pipeline. See CITATIONS.md for feature and CV references.

Only competition-provided train/test/sample files are read. Audit labels are
reserved until evaluate_audit is explicitly enabled. No competition upload occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

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
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "optuna", "joblib")
GROUPS = {"scores", "affordability", "tenure", "missingness", "peer", "categories", "logs"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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
        if set(self.models) - {"lightgbm", "catboost"}:
            raise ValueError("This first pipeline supports LightGBM and CatBoost")
        if set(self.feature_groups) - GROUPS:
            raise ValueError("Unknown feature group")
        if not 0 < self.audit_fraction < 0.5 or self.n_splits < 2:
            raise ValueError("Invalid validation split")
        if self.mode == "final" and not self.frozen_choices:
            raise ValueError("Set frozen_choices=True only after model/feature selection is finished")
        if self.mode == "final" and not self.blend_weights:
            raise ValueError("Final mode requires the previously selected blend_weights")
        if self.mode == "final" and self.evaluate_audit:
            raise ValueError("Final training uses former audit rows; an audit score would not be independent")
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
        return self

    def transform(self, x):
        z = x.copy()
        for c, mapping in self.peers.items():
            peer = z[c].fillna("__MISSING__").map(mapping).fillna(self.income_median)
            z[f"{c}_PEER_INCOME"] = peer
            z[f"{c}_RELATIVE_INCOME"] = safe_ratio(z.AMT_INCOME_TOTAL, peer)
        for c, mapping in self.frequencies.items():
            z[f"{c}_FREQUENCY"] = z[c].fillna("__MISSING__").map(mapping).fillna(0)
        for c in self.categorical:
            if self.model_name == "lightgbm":
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
    return dict(loss_function="Logloss", eval_metric="AUC", iterations=cfg.max_iterations,
                learning_rate=0.04, depth=6, l2_leaf_reg=6, random_seed=cfg.seed,
                thread_count=cfg.threads, allow_writing_files=False, verbose=False)


def fit_cv(x, y, test_x, ids, cfg, output_dir=None, audit_x=None, trial=None):
    result = {"oof": {}, "test": {}, "audit": {}, "metrics": {}}
    folds = list(StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed).split(x, y))
    fold_ids = np.full(len(x), -1, dtype=int)
    for fold, (_, va) in enumerate(folds):
        fold_ids[va] = fold
    result["fold_ids"] = fold_ids
    for name in cfg.models:
        oof = np.full(len(x), np.nan)
        test_pred = np.zeros(len(test_x)) if test_x is not None else None
        audit_pred = np.zeros(len(audit_x)) if audit_x is not None else None
        fold_metrics = []
        for fold, (tr, va) in enumerate(folds):
            started = time.monotonic()
            processor = FeatureProcessor(cfg.feature_groups, name).fit(x.iloc[tr])
            train_x, valid_x = processor.transform(x.iloc[tr]), processor.transform(x.iloc[va])
            params = {**model_defaults(name, cfg), **cfg.model_params.get(name, {})}
            if name == "lightgbm":
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**params)
                model.fit(train_x, y.iloc[tr], eval_set=[(valid_x, y.iloc[va])],
                          callbacks=[lgb.early_stopping(cfg.early_stopping, first_metric_only=True, verbose=False)])
                best_iteration = int(model.best_iteration_ or cfg.max_iterations)
            else:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(**params)
                model.fit(train_x, y.iloc[tr], cat_features=processor.categorical,
                          eval_set=(valid_x, y.iloc[va]), early_stopping_rounds=cfg.early_stopping, use_best_model=True)
                best_iteration = int(model.tree_count_)
            oof[va] = model.predict_proba(valid_x)[:, 1]
            if test_pred is not None:
                test_pred += model.predict_proba(processor.transform(test_x))[:, 1] / cfg.n_splits
            if audit_pred is not None:
                audit_pred += model.predict_proba(processor.transform(audit_x))[:, 1] / cfg.n_splits
            metric = {"fold": fold, "auc": float(roc_auc_score(y.iloc[va], oof[va])),
                      "best_iteration": best_iteration, "seconds": round(time.monotonic() - started, 2)}
            fold_metrics.append(metric)
            print(f"{name} fold {fold + 1}/{cfg.n_splits}: AUC={metric['auc']:.6f}, trees={best_iteration}", flush=True)
            if output_dir is not None:
                joblib.dump({"processor": processor, "model": model}, Path(output_dir) / f"{name}_fold_{fold}.joblib")
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
        result["metrics"][name] = {"oof_auc": float(roc_auc_score(y, oof)), "folds": fold_metrics}
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

| Model | Development OOF AUC |
| --- | --- |
{rows}

Selected OOF AUC: **{manifest['selected_oof_auc']:.6f}**. This is a local model-selection result, not a leaderboard score. {audit_text}

- Feature groups: {', '.join(cfg['feature_groups']) or 'raw predictors + employment placeholder flag'}
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
    x = make_features(train.iloc[selected_rows].reset_index(drop=True), cfg.feature_groups)
    y = train.TARGET.iloc[selected_rows].reset_index(drop=True)
    ids = train.SK_ID_CURR.iloc[selected_rows].reset_index(drop=True)
    test_x = make_features(test, cfg.feature_groups)
    audit_x = make_features(train.iloc[audit].reset_index(drop=True), cfg.feature_groups) if cfg.evaluate_audit else None
    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%S%fZ") + "_" + cfg.label
    output_dir = Path(output_dir)
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", asdict(cfg))
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
    manifest = {"pipeline_version": PIPELINE_VERSION, "pipeline_sha256": sha256(__file__),
                "run_id": run_id, "created_utc": created.isoformat(), "config": asdict(cfg),
                "input_hashes": hashes, "package_versions": package_versions,
                "training_rows": len(train), "development_rows": len(development), "audit_rows": len(audit),
                "fitted_population_rows": len(selected_rows), "test_rows": len(test), "feature_count": x.shape[1],
                "model_metrics": results["metrics"], "selected_weights": weights, "selected_name": selected_name,
                "selected_oof_auc": float(roc_auc_score(y, selected_oof)), "audit_auc": audit_auc,
                "submission_sha256": sha256(run_dir / "submission.csv"), "competition_upload": False,
                "seconds": round(time.monotonic() - started, 2), "run_dir": str(run_dir.resolve())}
    save_json(run_dir / "manifest.json", manifest)
    (run_dir / "requirements.lock.txt").write_text("\n".join(f"{name}=={value}" for name, value in package_versions.items()) + "\n")
    shutil.copy2(run_dir / "submission.csv", output_dir / "submission.csv")
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


def tune_model(input_dir, study_dir, cfg, model_name="lightgbm", n_trials=20, timeout=3600):
    """Persistent seeded Optuna search on development rows only."""
    import optuna
    if cfg.mode != "development" or cfg.evaluate_audit:
        raise ValueError("Tuning is restricted to development data; disable audit evaluation")
    train, _, _, hashes = load_inputs(input_dir, cfg.verify_input_hashes)
    dev, _ = development_split(train, cfg)
    x = make_features(train.iloc[dev].reset_index(drop=True), cfg.feature_groups)
    y = train.TARGET.iloc[dev].reset_index(drop=True)
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    identity = {"input_hashes": hashes, "groups": cfg.feature_groups, "model": model_name,
                "seed": cfg.seed, "audit_seed": cfg.audit_seed, "audit_fraction": cfg.audit_fraction,
                "max_iterations": cfg.max_iterations, "early_stopping": cfg.early_stopping,
                "model_params": cfg.model_params, "versions": {p: version(p) for p in PACKAGES}}
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
        else:
            raise ValueError("Unsupported tuning model")
        trial_cfg = Config(**{**asdict(cfg), "models": (model_name,), "n_splits": 3,
                              "model_params": {model_name: params}, "blend_weights": None})
        fitted = fit_cv(x, y, None, train.SK_ID_CURR.iloc[dev], trial_cfg, trial=trial)
        return fitted["metrics"][model_name]["oof_auc"]

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
    best = {**study.best_params}
    if model_name == "lightgbm":
        best["subsample_freq"] = 1
    save_json(study_dir / f"{study_name}_best.json", {"params": best, "selection_auc": study.best_value, "identity": identity})
    study.trials_dataframe().to_csv(study_dir / f"{study_name}_trials.csv", index=False)
    return best


def select_blend(run_dir, seed=42):
    """OOF meta-holdout diagnostic. Independent audit confirmation still needed."""
    run_dir = Path(run_dir)
    oof = pd.read_csv(run_dir / "oof.csv")
    if not {"lightgbm", "catboost"}.issubset(oof.columns):
        raise ValueError("Both model OOF columns are required")
    fit, validation = train_test_split(np.arange(len(oof)), stratify=oof.TARGET, test_size=0.30, random_state=seed + 17)
    rows = []
    for weight in np.linspace(0, 1, 11):
        p = weight * oof.lightgbm + (1 - weight) * oof.catboost
        rows.append({"lightgbm_weight": float(weight), "selection_auc": float(roc_auc_score(oof.TARGET.iloc[fit], p.iloc[fit])),
                     "meta_validation_auc": float(roc_auc_score(oof.TARGET.iloc[validation], p.iloc[validation]))})
    table = pd.DataFrame(rows)
    best = table.sort_values("selection_auc", ascending=False).iloc[0]
    table.to_csv(run_dir / "blend_candidates.csv", index=False)
    weights = {"lightgbm": float(best.lightgbm_weight), "catboost": float(1 - best.lightgbm_weight)}
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
    items = {"comp.ipynb": notebook, "gci_pipeline.py": Path(__file__),
             "reproduce_config.json": run_dir / "reproduce_config.json", "manifest.json": run_dir / "manifest.json",
             "requirements.txt": run_dir / "requirements.lock.txt", "submission.csv": run_dir / "submission.csv"}
    readme = """# Reproduce this GCI submission\n\nPython 3.12 is recommended. Install requirements.txt.\nPut the three competition-provided CSVs in input/. Then run:\n\npython gci_pipeline.py --input input --output reproduced --config reproduce_config.json\n\nThis command uses the executed pipeline and frozen configuration. Compare reproduced/submission.csv with the bundled CSV. Byte identity can depend on platform/library versions. The manifest records the original hash. comp.ipynb offers the Colab interface. No external data or manual overrides are used. No Omnicampus upload is automated.\n"""
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in items.items():
            archive.write(path, arcname=name)
        archive.writestr("README.md", readme)
        citations_path = project_dir / "CITATIONS.md"
        if citations_path.exists():
            archive.write(citations_path, "CITATIONS.md")
        else:
            raise FileNotFoundError("CITATIONS.md must accompany the code ZIP")
        archive.writestr("bundle_hashes.json", json.dumps({name: sha256(path) for name, path in items.items()}, indent=2))
    with zipfile.ZipFile(bundle_path) as archive:
        if archive.testzip() is not None or hashlib.sha256(archive.read("submission.csv")).hexdigest() != manifest["submission_sha256"]:
            raise ValueError("ZIP integrity check failed")
    shutil.copy2(bundle_path, run_dir / "comp.zip")
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
