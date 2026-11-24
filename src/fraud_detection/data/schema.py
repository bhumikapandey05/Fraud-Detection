"""The expected structure of the Kaggle credit card fraud dataset.

Keeping the schema in one place means the loader, the validator and the
preprocessing code all agree on what the columns are.
"""

from __future__ import annotations

PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
TARGET_COLUMN = "Class"

FEATURE_COLUMNS: list[str] = [TIME_COLUMN, *PCA_FEATURES, AMOUNT_COLUMN]
EXPECTED_COLUMNS: list[str] = [*FEATURE_COLUMNS, TARGET_COLUMN]

# float64 keeps the published precision of the PCA components; the target is a 0/1 int.
COLUMN_DTYPES: dict[str, str] = {col: "float64" for col in FEATURE_COLUMNS}
COLUMN_DTYPES[TARGET_COLUMN] = "int64"

VALID_TARGET_VALUES: frozenset[int] = frozenset({0, 1})

# The dataset covers two days of transactions (Time is seconds since the first one).
MAX_EXPECTED_TIME_SECONDS = 2 * 24 * 3600
