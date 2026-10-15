"""
Assignment 1 — SELF-CHECK (a direction, not a grade).

Run:   python hw1_self_check.py hw1_<ID>.py
(keep hw1_measurements.csv and hw1_labels.csv in the same folder)

What it checks
  1. Format      — the grader can import your file and find every name.
  2. Sanity      — your outputs have the right shape and no obvious contradictions
                   (no sentinels left, plausible alcohol values, no duplicate ids,
                   no missing values after imputation, correct pipeline steps, ...).
  3. Part B      — the constants you typed agree with what YOUR OWN functions compute.

What it does NOT check
  * whether your results are the true answers (the grader also uses hidden data);
  * Part C.
All checks passing is a good sign — not a guarantee of full marks.
"""
import importlib.util
import os
import re
import signal
import sys
import time
import warnings

FUNCTIONS = ["clean_measurements", "merge_labels", "split_data", "impute_with_train_median",
             "make_logreg", "best_C_by_cv", "count_nonzero_coefs", "components_for_variance",
             "pca_reconstruction_mse", "make_pca_knn"]
INTS = ["N_RAW_ROWS", "N_DUPLICATES_REMOVED", "N_FRACTION_ALCOHOL", "N_LABELED_SAMPLES",
        "N_MISSING_CELLS", "N_NONZERO_L1", "N90_SCALED", "N90_UNSCALED"]
FLOATS = ["BEST_C_L1", "L1_TEST_ACCURACY", "PCA_KNN_TEST_ACCURACY"]
LETTERS = [f"C{i}" for i in range(1, 11)]
TIME_LIMIT = 120
passed = warned = 0


def ok(msg):
    global passed
    passed += 1
    print("  ✔", msg)


def bad(msg):
    global warned
    warned += 1
    print("  ✘", msg)


def _timeout(signum, frame):
    raise TimeoutError()


def check(label, test):
    if hasattr(signal, "SIGALRM"):
        signal.alarm(TIME_LIMIT)
    try:
        result = test()
        ok(label) if result is True else bad(label + ("" if result in (False, None) else f" — {result}"))
    except NotImplementedError:
        bad(label + " — not implemented yet")
    except TimeoutError:
        bad(label + f" — took more than {TIME_LIMIT} s")
    except Exception as exc:
        bad(label + f" — {type(exc).__name__}: {str(exc)[:120]}")
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


def main(path):
    warnings.filterwarnings("ignore")
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout)
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n1. Format")
    ok("file name") if re.fullmatch(r"hw1_\d{5,10}(_\d{5,10})?\.py", os.path.basename(path)) else \
        bad("file name should be hw1_<ID>.py or hw1_<ID1>_<ID2>.py")
    start = time.time()
    try:
        spec = importlib.util.spec_from_file_location("submission", path)
        sub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sub)
    except Exception as exc:
        bad(f"the file cannot be imported: {type(exc).__name__}: {exc}")
        return summary()
    warnings.filterwarnings("ignore")
    ok("the file imports") if time.time() - start < 30 else \
        bad("importing takes > 30 s — move experiments under if __name__ == '__main__':")
    ids = getattr(sub, "STUDENT_IDS", None)
    ok("STUDENT_IDS") if isinstance(ids, (list, tuple)) and 1 <= len(ids) <= 2 and \
        all(isinstance(x, str) and re.fullmatch(r"\d{5,10}", x) for x in ids) else \
        bad('STUDENT_IDS must be a list of 1–2 ID strings, e.g. ["123456789"]')
    missing = [f for f in FUNCTIONS if not callable(getattr(sub, f, None))]
    ok("all functions exist") if not missing else bad(f"missing functions: {missing}")
    wrong = [n for n in INTS if not (isinstance(getattr(sub, n, None), int) and not isinstance(getattr(sub, n), bool))]
    wrong += [n for n in FLOATS if not isinstance(getattr(sub, n, None), (int, float)) or isinstance(getattr(sub, n), bool)]
    if not isinstance(getattr(sub, "TOP_FEATURE_PC1_UNSCALED", None), str):
        wrong.append("TOP_FEATURE_PC1_UNSCALED")
    wrong += [n for n in LETTERS if not (isinstance(getattr(sub, n, None), str)
                                         and getattr(sub, n).strip().lower() in {"a", "b", "c", "d"})]
    ok("constants have the right types") if not wrong else bad(f"not filled in / wrong type: {', '.join(wrong)}")

    print("\n2. Sanity checks on the course data")
    folder = os.path.dirname(os.path.abspath(path))
    mp, lp = os.path.join(folder, "hw1_measurements.csv"), os.path.join(folder, "hw1_labels.csv")
    if not (os.path.exists(mp) and os.path.exists(lp)):
        bad("hw1_measurements.csv / hw1_labels.csv not found next to your file — sections 2–3 skipped")
        return summary()
    meas, labels = pd.read_csv(mp), pd.read_csv(lp)
    F = sub.FEATURES
    st = {}

    def t_clean():
        before = meas.copy()
        c = sub.clean_measurements(meas)
        st["c"] = c
        probs = []
        if not meas.equals(before):
            probs.append("the input was modified")
        if list(c.columns) != ["sample_id"] + F:
            probs.append("columns should be ['sample_id'] + FEATURES")
        if not c["sample_id"].astype(str).str.fullmatch(r"W\d{4}").all():
            probs.append("some sample_id values are not normalized")
        if c["sample_id"].duplicated().any():
            probs.append("repeated sample_id values remain")
        if not all(pd.api.types.is_float_dtype(c[f]) for f in F):
            probs.append("some FEATURES columns are not numeric")
        elif (c[F] < 0).any().any():
            probs.append("negative values remain (a missing-value code?)")
        elif not c["alcohol"].dropna().between(5, 25).all():
            probs.append("some alcohol values are not in percent")
        if list(c.index) != list(range(len(c))):
            probs.append("index is not 0..n-1")
        return True if not probs else "; ".join(probs)
    check("clean_measurements", t_clean)
    if "c" not in st:
        return summary()

    def t_merge():
        d = sub.merge_labels(st["c"].copy(), labels)
        st["d"] = d
        probs = []
        if list(d.columns) != ["sample_id"] + F + ["cultivar"]:
            probs.append("columns should be ['sample_id'] + FEATURES + ['cultivar']")
        if d["cultivar"].isna().any():
            probs.append("some samples have no cultivar (should be an inner join)")
        if sorted(d["cultivar"].dropna().unique()) != ["Barbera", "Barolo", "Grignolino"]:
            probs.append(f"cultivar values should be exactly Barbera, Barolo, Grignolino "
                         f"(found {sorted(map(str, d['cultivar'].unique()))[:6]})")
        if not set(d["sample_id"]) <= set(st["c"]["sample_id"]):
            probs.append("samples without measurements were kept")
        return True if not probs else "; ".join(probs)
    check("merge_labels", t_merge)
    if "d" not in st:
        return summary()

    def t_split():
        Xtr, Xte, ytr, yte = sub.split_data(st["d"])
        st.update(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
        n = len(st["d"])
        if len(Xte) != int(np.ceil(0.25 * n)) or len(Xtr) + len(Xte) != n:
            return "the test part should be 25% of the samples"
        return list(Xtr.columns) == F or "X should contain exactly the FEATURES columns"
    check("split_data", t_split)
    if "Xtr" not in st:
        return summary()

    def t_impute():
        a, b = st["Xtr"].copy(), st["Xte"].copy()
        tr, te = sub.impute_with_train_median(a, b)
        st.update(Xtr_f=tr, Xte_f=te)
        if tr.isna().any().any() or te.isna().any().any():
            return "missing values remain"
        if not (a.equals(st["Xtr"]) and b.equals(st["Xte"])):
            return "the inputs were modified"
        med = st["Xtr"].median()
        filled = st["Xte"].isna()
        return all(np.allclose(te.loc[filled[c], c], med[c]) for c in F if filled[c].any()) or \
            "test values were not filled with the TRAINING medians"
    check("impute_with_train_median (no missing values left, training medians)", t_impute)
    if "Xtr_f" not in st:
        return summary()
    Xtr, Xte, ytr, yte = st["Xtr_f"], st["Xte_f"], st["ytr"], st["yte"]

    def t_logreg():
        p = sub.make_logreg(0.5, "l1")
        steps = dict(p.named_steps) if isinstance(p, Pipeline) else {}
        if list(steps) != ["scale", "model"]:
            return "steps should be named 'scale' and 'model'"
        m = steps["model"]
        return (isinstance(steps["scale"], StandardScaler) and isinstance(m, LogisticRegression)
                and m.C == 0.5 and m.penalty == "l1" and m.solver == "saga") or "wrong step settings"
    check("make_logreg structure", t_logreg)

    def t_bestC():
        c = sub.best_C_by_cv(Xtr, ytr, sub.C_CANDIDATES, "l1")
        st["bestC"] = float(c)
        return float(c) in [float(x) for x in sub.C_CANDIDATES] or "the result is not one of the candidates"
    check("best_C_by_cv returns one of the candidates", t_bestC)

    def t_nonzero():
        m = sub.make_logreg(st.get("bestC", 1.0), "l1").fit(Xtr, ytr)
        st["l1"] = m
        n = sub.count_nonzero_coefs(m)
        st["nz"] = n
        return (isinstance(n, (int, np.integer)) and 0 < n <= 39) or "should be an int between 1 and 39"
    check("count_nonzero_coefs", t_nonzero)

    def t_comp():
        s1 = sub.components_for_variance(Xtr, 0.90, True)
        s0 = sub.components_for_variance(Xtr, 0.90, False)
        s99 = sub.components_for_variance(Xtr, 0.99, True)
        st.update(n90s=s1, n90u=s0)
        if not all(isinstance(v, (int, np.integer)) and 1 <= v <= 13 for v in (s1, s0, s99)):
            return "results should be ints between 1 and 13"
        return s99 >= s1 or "a higher threshold cannot need fewer components"
    check("components_for_variance (consistent thresholds)", t_comp)

    def t_recon():
        errs = [sub.pca_reconstruction_mse(Xtr, Xte, k) for k in (2, 5, 13)]
        if not errs[0] >= errs[1] >= errs[2] - 1e-9:
            return "the error should not grow when more components are kept"
        return errs[2] < 1e-6 or "with all 13 components the reconstruction should be exact"
    check("pca_reconstruction_mse (decreases with k, ~0 with all components)", t_recon)

    def t_pcaknn():
        p = sub.make_pca_knn(0.85, 7)
        steps = dict(p.named_steps) if isinstance(p, Pipeline) else {}
        if list(steps) != ["scale", "pca", "knn"]:
            return "steps should be named 'scale', 'pca' and 'knn'"
        good = (isinstance(steps["scale"], StandardScaler) and isinstance(steps["pca"], PCA)
                and steps["pca"].n_components == 0.85 and isinstance(steps["knn"], KNeighborsClassifier)
                and steps["knn"].n_neighbors == 7)
        st["knn"] = sub.make_pca_knn(0.90, 5).fit(Xtr, ytr).score(Xte, yte)
        return good or "wrong step settings"
    check("make_pca_knn structure", t_pcaknn)

    print("\n3. Part B constants vs YOUR functions (consistency, not correctness)")
    sid = meas["sample_id"].astype(str).str.strip().str.upper()
    mine = {"N_RAW_ROWS": len(meas), "N_DUPLICATES_REMOVED": len(meas) - len(st["c"]),
            "N_FRACTION_ALCOHOL": int((meas.loc[~sid.duplicated(), "alcohol_unit"] == "fraction").sum()),
            "N_LABELED_SAMPLES": len(st["d"]),
            "N_MISSING_CELLS": int(st["d"][F].isna().sum().sum())}
    if "bestC" in st:
        mine["BEST_C_L1"] = st["bestC"]
    if "l1" in st:
        mine["N_NONZERO_L1"] = st["nz"]
        mine["L1_TEST_ACCURACY"] = float(st["l1"].score(Xte, yte))
    if "n90s" in st:
        mine["N90_SCALED"], mine["N90_UNSCALED"] = st["n90s"], st["n90u"]
    mine["TOP_FEATURE_PC1_UNSCALED"] = F[int(np.argmax(np.abs(PCA().fit(Xtr).components_[0])))]
    if "knn" in st:
        mine["PCA_KNN_TEST_ACCURACY"] = float(st["knn"])
    for key, value in mine.items():
        typed = getattr(sub, key, None)
        if typed is None:
            bad(f"{key} is not filled in")
        elif isinstance(value, str):
            ok(key) if str(typed).strip().lower() == value else bad(f"{key} differs from what your data give")
        elif key in ("L1_TEST_ACCURACY", "PCA_KNN_TEST_ACCURACY"):
            ok(key) if abs(float(typed) - value) <= 0.0015 else \
                bad(f"{key} differs from what your code computes (check rounding to 3 decimals)")
        else:
            ok(key) if float(typed) == float(value) else bad(f"{key} differs from what your code computes")
    print("  –  Part C is not checked here.")
    return summary()


def summary():
    print(f"\n{passed} passed, {warned} to look at.")
    print("Passing every check is a good direction, not a guarantee: the grader also uses hidden data.")
    return 0 if warned == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))