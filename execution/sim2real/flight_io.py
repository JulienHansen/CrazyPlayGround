"""One reader and one writer for flight recordings, shared by every tool here.

Parquet is the default format. On a real 1251-step flight it is 4.1x smaller than
the equivalent CSV (232 KB vs 943 KB) and reads 8x faster, and it carries dtypes
so a column cannot silently come back as a string. CSV stays available because a
flight laptop in the lab may not have pyarrow, and because a CSV can be inspected
with nothing but a text editor between two flights.

Readers accept either format, so recordings made before the switch keep working.
"""

import csv
import os

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:                                    # deployment laptop without pyarrow
    pa = pq = None

PARQUET_NAME = "flight.parquet"
CSV_NAME = "flight.csv"
COMPRESSION = "zstd"


def parquet_available() -> bool:
    return pq is not None


def write_flight(run_dir: str, records, columns, prefer_parquet: bool = True,
                 also_csv: bool = False) -> str:
    """Write one row per control step. Returns the path actually written.

    Falls back to CSV when pyarrow is missing: losing a flight because a library
    is absent is never the right trade.
    """
    os.makedirs(run_dir, exist_ok=True)
    wrote = None
    if prefer_parquet and pq is not None:
        path = os.path.join(run_dir, PARQUET_NAME)
        table = pa.table({c: pa.array([r[c] for r in records], type=pa.float64())
                          for c in columns})
        pq.write_table(table, path, compression=COMPRESSION)
        wrote = path
    if also_csv or wrote is None:
        path = os.path.join(run_dir, CSV_NAME)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(records)
        wrote = wrote or path
    return wrote


def resolve(path: str) -> str:
    """Return the recording file for a run directory, whichever format it is in."""
    if not os.path.isdir(path):
        return path
    for name in (PARQUET_NAME, CSV_NAME):
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(path, CSV_NAME)          # report the CSV name when neither exists


def read_flight(path: str) -> dict:
    """Load a recording as {column: float64 array}, from Parquet or CSV."""
    path = resolve(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No flight recording at {path}")
    if path.endswith(".parquet"):
        if pq is None:
            raise RuntimeError(f"{path} is Parquet but pyarrow is not installed")
        table = pq.read_table(path)
        return {name: np.asarray(table[name].to_numpy(), dtype=np.float64)
                for name in table.column_names}
    cols = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for name in reader.fieldnames:
            cols[name] = []
        for row in reader:
            for name in reader.fieldnames:
                cols[name].append(float(row[name]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
