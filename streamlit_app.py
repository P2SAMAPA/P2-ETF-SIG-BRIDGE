import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfApi
from datetime import date, timedelta
import config
import os

st.set_page_config(page_title="Sig-Bridge Engine", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2.3rem;font-weight:700;color:#1a1a2e;margin-bottom:0.2rem}
.sub-header{font-size:1rem;color:#555;margin-bottom:1.5rem}
.uni-title{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
           padding-left:0.5rem;border-left:5px solid #e94560}
.hero-card{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(233,69,96,0.3)}
.win-card{background:linear-gradient(135deg,#0f3460 0%,#533483 100%);
          color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
          box-shadow:0 4px 12px rgba(83,52,131,0.3)}
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.badge-buy{background:#27ae60;border-radius:6px;padding:2px 12px;font-size:0.75rem;
           font-weight:700;color:white}
.badge-sell{background:#e74c3c;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-hold{background:#f39c12;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-strongbuy{background:#1a7a2a;border-radius:6px;padding:2px 12px;font-size:0.75rem;
                 font-weight:700;color:white}
.badge-strongsell{background:#7a1a1a;border-radius:6px;padding:2px 12px;font-size:0.75rem;
                  font-weight:700;color:white}
.badge-bridge{background:#8e44ad;border-radius:6px;padding:2px 8px;font-size:0.65rem;
              font-weight:700;color:white}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌉 Sig-Bridge Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Signature-Conditioned Neural Bridge · Schrödinger Bridge · '
    'Path Signatures · Conditional Flow Matching</div>',
    unsafe_allow_html=True)

HF_TOKEN = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
RESULTS_REPO = config.RESULTS_REPO

US_HOLIDAYS = {
    date(2025,1,1),date(2025,1,20),date(2025,2,17),date(2025,4,18),
    date(2025,5,26),date(2025,6,19),date(2025,7,4),date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def next_trading_day() -> str:
    d = date.today() + timedelta(days
