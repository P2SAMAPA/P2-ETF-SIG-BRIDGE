# P2-SIG-BRIDGE

**Signature-Conditioned Neural Bridge — Neural SDEs with Path Signatures**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine solves the **Schrödinger Bridge problem** using **signature-conditioned neural SDEs**. It generates scenario paths that interpolate between two known market states, conditioned on the truncated path signature.

### Theory

**Schrödinger Bridge:** The most likely path between two distributions, balancing the prior process and observed endpoints.

**Path Signature:** The iterated integrals of a path, capturing its entire history as a feature vector.

**Conditional Flow Matching:** Learns the drift of the neural SDE by matching conditional probability flows.

**Key Insight:** The bridge is conditioned on the **entire path history** (via its signature), not just the final state.

---

## Key Metrics

| Metric | What it tells you |
|--------|-------------------|
| **z-score** | Cross-sectional ranking of bridge complexity |
| **Bridge Width** | Uncertainty in the path interpolation |
| **Bridge Curvature** | Non-linearity of the bridge |
| **N Paths** | Number of generated bridge paths |

---

## Windows

| Window | Purpose |
|--------|---------|
| 21d | Ultra-short bridge |
| 63d | Core signal (primary) |
| 126d | Medium-term bridge |
| 252d | Structural bridge |
| 504d | Long-term bridge |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-SIG-BRIDGE
cd P2-SIG-BRIDGE
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
