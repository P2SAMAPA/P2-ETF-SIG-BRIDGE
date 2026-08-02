"""
config.py  —  Configuration for Sig-Bridge Engine
==================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - SIGNATURE: Path signature parameters
  - NEURAL_SDE: Neural SDE parameters
  - BRIDGE: Schrödinger Bridge parameters
  - WINDOWS: Time windows for bridge construction
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-sig-bridge-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [21, 63, 126, 252, 504]
WINDOW_LABELS = {
    21: "21d  (~1 month) — Short-term",
    63: "63d  (~3 months) — Core Signal",
    126: "126d (~6 months) — Medium-term",
    252: "252d (~1 year) — Structural",
    504: "504d (~2 years) — Long-term",
}
PRIMARY_WINDOW = 63


# ── Signature Parameters ──────────────────────────────────────────────────

SIGNATURE = {
    "depth": 3,
    "n_landmarks": 100,
    "include_time": True,
    "normalize": True,
}


# ── Neural SDE Parameters ──────────────────────────────────────────────────

NEURAL_SDE = {
    "state_dim": 16,
    "hidden_dim": 128,
    "n_layers": 3,
    "learning_rate": 0.001,
    "n_epochs": 100,
    "batch_size": 64,
}


# ── Schrödinger Bridge Parameters ──────────────────────────────────────────

BRIDGE = {
    "n_time_steps": 30,
    "n_paths": 100,
    "flow_matching": True,
    "temperature": 1.0,
    "convergence_threshold": 1e-4,
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
