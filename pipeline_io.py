"""
pipeline_io.py — IO manager terpusat untuk pipeline NB1-NB6.

Setiap notebook punya 'stage' yang isolasi outputnya:
    NB1 : stage='01_well_qc'
    NB2 : stage='02_well_cond'
    NB3 : stage='03_seismic'
    NB4 : stage='04_tie'
    NB5 : stage='05_lfm'
    NB6 : stage='06_inversion'

Folder default: ./pipeline_data/<stage>/
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class PipelineIO:
    """Per-stage IO. Supports DataFrame (parquet/csv), pickle, npz, json, joblib."""

    def __init__(self, stage: str, root: str | Path = "pipeline_data"):
        self.stage = stage
        self.root = Path(root)
        self.dir = self.root / stage
        self.dir.mkdir(parents=True, exist_ok=True)

    # ─────────── DataFrame ───────────
    def save_df(self, name: str, df: pd.DataFrame) -> Path:
        """Save DataFrame as parquet (preferred) or csv fallback."""
        try:
            p = self.dir / f"{name}.parquet"
            df.to_parquet(p, index=False)
        except Exception:
            p = self.dir / f"{name}.csv"
            df.to_csv(p, index=False)
        return p

    def load_df(self, name: str) -> pd.DataFrame:
        for ext in ("parquet", "csv"):
            p = self.dir / f"{name}.{ext}"
            if p.exists():
                return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
        raise FileNotFoundError(f"DataFrame '{name}' tidak ditemukan di {self.dir}")

    # ─────────── Pickle ───────────
    def save_pkl(self, name: str, obj: Any) -> Path:
        p = self.dir / f"{name}.pkl"
        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    def load_pkl(self, name: str) -> Any:
        p = self.dir / f"{name}.pkl"
        with open(p, "rb") as f:
            return pickle.load(f)

    # ─────────── NPZ ───────────
    def save_npz(self, name: str, **arrays) -> Path:
        p = self.dir / f"{name}.npz"
        np.savez_compressed(p, **arrays)
        return p

    def load_npz(self, name: str) -> Dict[str, np.ndarray]:
        p = self.dir / f"{name}.npz"
        with np.load(p, allow_pickle=True) as f:
            return {k: f[k] for k in f.files}

    # ─────────── JSON ───────────
    def save_json(self, name: str, obj: Any) -> Path:
        p = self.dir / f"{name}.json"
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        return p

    def load_json(self, name: str) -> Any:
        with open(self.dir / f"{name}.json") as f:
            return json.load(f)

    # ─────────── Joblib (untuk sklearn objek) ───────────
    def save_joblib(self, name: str, obj: Any) -> Path:
        if not HAS_JOBLIB:
            return self.save_pkl(name, obj)
        p = self.dir / f"{name}.joblib"
        joblib.dump(obj, p)
        return p

    def load_joblib(self, name: str) -> Any:
        if not HAS_JOBLIB:
            return self.load_pkl(name)
        p = self.dir / f"{name}.joblib"
        if p.exists():
            return joblib.load(p)
        return self.load_pkl(name)

    # ─────────── Cross-stage loader ───────────
    @staticmethod
    def from_stage(stage: str, root: str | Path = "pipeline_data") -> "PipelineIO":
        """Untuk load dari stage lain tanpa create instance default."""
        return PipelineIO(stage=stage, root=root)

    # ─────────── Utility ───────────
    def list_files(self) -> None:
        files = sorted(self.dir.iterdir())
        if not files:
            print(f"  [{self.stage}] (empty)")
            return
        print(f"  [{self.stage}] {len(files)} file(s) di {self.dir}:")
        for f in files:
            kb = f.stat().st_size / 1024
            print(f"    • {f.name:40s}  {kb:>8.1f} KB")

    def exists(self, name: str) -> bool:
        for ext in ("parquet", "csv", "pkl", "npz", "json", "joblib"):
            if (self.dir / f"{name}.{ext}").exists():
                return True
        return False
