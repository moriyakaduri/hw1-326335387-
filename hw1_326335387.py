"""
Data Mining — Assignment 1: Preprocessing, Regularization and PCA
=================================================================

Fill in every function and every constant marked TODO, then submit this single
file to Moodle, renamed to  hw1_<ID>.py  (for a pair: hw1_<ID1>_<ID2>.py).

RULES — read before you start
-----------------------------
- The file is graded automatically AFTER submission. There is no correctness
  check before the deadline; hw1_self_check.py only gives you a direction.
- Do not rename functions, parameters or constants, and keep the return types
  described in each docstring. A wrong name or type gets no points.
- Allowed imports: the Python standard library, numpy, pandas and scikit-learn.
- Code that is not inside a function must be placed under
      if __name__ == "__main__":
  The grader imports your file; long top-level code can make the import fail.
- Every function is tested on hidden data sets (other samples with the same kinds
  of problems) in addition to the course data. Hard-coding results does not work.
- Never fit anything (scaler, imputer, PCA, model) on test data.

The story
---------
A wine laboratory measured 13 chemical properties of wine samples from three
cultivars (Barolo, Grignolino, Barbera). The measurements and the cultivar labels
arrive in two separate files, with several data-quality problems:

    hw1_measurements.csv — sample_id, alcohol, alcohol_unit, 12 more measurements
    hw1_labels.csv       — sample_id, cultivar

Known problems (all of them occur, possibly also in the hidden data):
- sample_id sometimes has surrounding spaces and/or lower-case letters;
- missing values are written as "?", as an empty cell, or as -1;
- alcohol is usually in percent, but when alcohol_unit == "fraction" it was
    recorded as a fraction (0.132 instead of 13.2);
- some rows appear twice (the same sample_id);
- cultivar names use inconsistent case and spaces ("  barolo ", "BAROLO");
- some samples have no label, and some labels have no sample.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

STUDENT_IDS = ["123456789"]  # TODO: your ID number(s) as strings, e.g. ["123456789"]

MEASUREMENTS_PATH = "hw1_measurements.csv"
LABELS_PATH = "hw1_labels.csv"
RANDOM_STATE = 0
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols",
    "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue",
    "od280_od315", "proline",
]
C_CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


# =============================================================================
# Part A — implementation (60 points)
# =============================================================================

# ----- Preprocessing -----------------------------------------------------------

def clean_measurements(measurements):
    """Return a cleaned COPY of the raw measurements DataFrame.            [8 points]"""
    df = measurements.copy()

    # 1. sample_id: strip spaces and convert to upper case
    df["sample_id"] = df["sample_id"].astype(str).str.strip().str.upper()

    # 2. Replace "?", "", and -1 with NaN and convert FEATURES to numeric (float)
    df[FEATURES] = df[FEATURES].replace({"?": np.nan, "": np.nan, -1: np.nan, "-1": np.nan})
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    # 3. Multiply alcohol by 100 where alcohol_unit == "fraction", then drop alcohol_unit
    mask = df["alcohol_unit"] == "fraction"
    df.loc[mask, "alcohol"] = df.loc[mask, "alcohol"] * 100
    df = df.drop(columns=["alcohol_unit"])

    # 4. Remove repeated samples keeping only the FIRST row
    df = df.drop_duplicates(subset=["sample_id"], keep="first")

    # Return specific columns with a fresh index
    df = df[["sample_id"] + FEATURES].reset_index(drop=True)
    return df


def merge_labels(measurements, labels):
    """Attach the cultivar label to every cleaned sample.                    [5 points]"""
    l_df = labels.copy()

    # Normalize labels.sample_id
    l_df["sample_id"] = l_df["sample_id"].astype(str).str.strip().str.upper()

    # Normalize labels.cultivar
    l_df["cultivar"] = l_df["cultivar"].astype(str).str.strip().str.title()

    # Inner join, maintaining measurements row order
    merged = pd.merge(measurements, l_df, on="sample_id", how="inner")

    return merged[["sample_id"] + FEATURES + ["cultivar"]].reset_index(drop=True)


def split_data(data):
    """Split the merged data into a development (train) part and a sealed test part.  [3 points]"""
    X = data[FEATURES]
    y = data["cultivar"]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)


def impute_with_train_median(X_train, X_test):
    """Fill missing values WITHOUT leakage.                                  [6 points]"""
    medians = X_train.median()
    X_train_filled = X_train.fillna(medians)
    X_test_filled = X_test.fillna(medians)
    return X_train_filled, X_test_filled


# ----- Regularization ---------------------------------------------------------------

def make_logreg(C, penalty):
    """Return an UNFITTED Pipeline with two steps:                           [4 points]"""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(penalty=penalty, C=C, solver="saga",
                                     max_iter=10000, random_state=RANDOM_STATE))
    ])


def best_C_by_cv(X, y, C_values, penalty):
    """Choose C by 5-fold cross-validation on (X, y) only.                  [8 points]"""
    pipeline = make_logreg(1.0, penalty)
    param_grid = {"model__C": sorted(C_values)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy")
    grid.fit(X, y)

    return float(grid.best_params_["model__C"])


def count_nonzero_coefs(fitted_pipeline, tol=1e-8):
    """Return the number of coefficients of the fitted "model" step whose absolute
    value is greater than tol (over all classes), as an int.             [4 points]"""
    model = fitted_pipeline.named_steps["model"]
    return int(np.sum(np.abs(model.coef_) > tol))


# ----- PCA ----------------------------------------------------------------------------

def components_for_variance(X, threshold, scale):
    """Smallest number of principal components whose cumulative explained variance
    ratio is >= threshold.                                                [8 points]"""
    if scale:
        X = StandardScaler().fit_transform(X)

    pca = PCA().fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cum_var >= threshold) + 1)


def pca_reconstruction_mse(X_train, X_test, n_components):
    """Reconstruction error of the TEST data in standardized units.        [7 points]"""
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components).fit(X_train_scaled)
    X_test_proj = pca.transform(X_test_scaled)
    X_test_recon = pca.inverse_transform(X_test_proj)

    mse = np.mean((X_test_scaled - X_test_recon) ** 2)
    return float(mse)


def make_pca_knn(variance_threshold, n_neighbors):
    """Return an UNFITTED Pipeline with three steps:                        [7 points]"""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=variance_threshold)),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))
    ])


# =============================================================================
# Part B — analysis of the course data (25 points)
# =============================================================================

# B1 [2] Number of rows in hw1_measurements.csv (before any cleaning).
N_RAW_ROWS = 186  # int

# B2 [2] Number of rows removed by clean_measurements because their sample_id repeats.
N_DUPLICATES_REMOVED = 8  # int

# B3 [2] Number of samples (after removing repeats) whose alcohol was recorded as a fraction.
N_FRACTION_ALCOHOL = 15  # int

# B4 [2] Number of samples in the merged data (merge_labels output).
N_LABELED_SAMPLES = 173  # int

# B5 [2] Number of missing cells in the FEATURES columns of the merged data (before imputation).
N_MISSING_CELLS = 79  # int

# B6 [3] best_C_by_cv(X_train_filled, y_train, C_CANDIDATES, "l1").
BEST_C_L1 = 0.3  # float

# B7 [2] Number of non-zero coefficients of make_logreg(BEST_C_L1, "l1") fitted on
#     (X_train_filled, y_train). The model has 3 x 13 = 39 coefficients.
N_NONZERO_L1 = 13  # int

# B8 [2] Test accuracy of that L1 model on (X_test_filled, y_test). Round to 3 decimals.
L1_TEST_ACCURACY = 0.977  # float

# B9 [2] components_for_variance(X_train_filled, 0.90, scale=True).
N90_SCALED = 8  # int

# B10 [2] components_for_variance(X_train_filled, 0.90, scale=False).
N90_UNSCALED = 1  # int

# B11 [2] Without scaling: the feature with the largest absolute weight (loading) in the
#      first principal component of PCA fitted on X_train_filled. A column name (str).
TOP_FEATURE_PC1_UNSCALED = "proline"  # str

# B12 [2] Test accuracy of make_pca_knn(0.90, 5) fitted on (X_train_filled, y_train).
#      Round to 3 decimals.
PCA_KNN_TEST_ACCURACY = 0.955  # float

# =============================================================================
# Part C — multiple-choice questions (15 points, 1.5 each)
# =============================================================================

C1 = "b"
C2 = "c"
C3 = "a"
C4 = "b"
C5 = "a"
C6 = "b"
C7 = "b"
C8 = "b"
C9 = "b"
C10 = "c"

if __name__ == "__main__":
    # Your own experiments go here. The grader does not run this block.
    meas = pd.read_csv(MEASUREMENTS_PATH)
    labels = pd.read_csv(LABELS_PATH)
    data = merge_labels(clean_measurements(meas), labels)
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_filled, X_test_filled = impute_with_train_median(X_train, X_test)
    print(data.shape, X_train_filled.shape, X_test_filled.shape)