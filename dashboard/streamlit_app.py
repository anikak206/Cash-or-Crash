"""
Cash or Crash - Decision Dashboard (Streamlit).

Run from the project root:
    pip install streamlit
    python -m streamlit run dashboard/streamlit_app.py

Shows, on the held-out TEST set:
  * how many defaulters are caught vs how many good customers are wrongly flagged
  * the simulated cost of those errors, and the cheapest threshold
  * the risk distribution and a single-customer lookup
Costs are ASSUMED example values, not real bank figures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `src` imports work

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.cost_analysis import best_row, evaluate_threshold, load_test_scores, sweep

st.set_page_config(page_title="Cash or Crash - Decision Dashboard", layout="wide")


@st.cache_data
def get_scores():
    y, proba = load_test_scores()
    return y, proba


y, proba = get_scores()

st.title("Cash or Crash - Adaptive Risk Decision Dashboard")
st.caption("Flag a customer as 'will default' when their predicted probability is at or above the threshold. "
           "Results are on 6,000 held-out test customers. Costs are assumed examples.")

# ---------------- sidebar controls ----------------
st.sidebar.header("Decision settings")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, float(config.LOW_RISK_MAX), 0.01)
st.sidebar.header("Assumed costs (per customer)")
cost_default = st.sidebar.number_input("Cost of a MISSED defaulter", 100, 100000, 2500, 100)
cost_alarm = st.sidebar.number_input("Cost of a FALSE alarm (good customer rejected)", 10, 100000, 500, 50)

now = evaluate_threshold(y, proba, threshold, cost_default, cost_alarm)
table = sweep(y, proba, cost_default, cost_alarm)
best = best_row(table)
approve_all = evaluate_threshold(y, proba, 1.01, cost_default, cost_alarm)

# ---------------- headline numbers ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Defaulters caught (recall)", f"{now['defaulters_caught_recall']:.1%}")
c2.metric("Customers flagged", f"{now['flagged_share']:.1%}")
c3.metric("Good customers wrongly flagged", f"{now['fp']:,}", f"{now['false_alarm_rate']:.1%} of good customers",
          delta_color="off")
c4.metric("Total cost", f"{now['total_cost']:,.0f}",
          f"{(now['total_cost'] - best['total_cost']):,.0f} above the best" if now['total_cost'] > best['total_cost']
          else "this is the best", delta_color="inverse")

tab1, tab2, tab3 = st.tabs(["Threshold vs cost", "Risk distribution", "Single customer"])

with tab1:
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(table["threshold"], table["total_cost"], color="black", label="Total cost")
        ax.plot(table["threshold"], table["cost_missed_defaulters"], "--", label="Missed defaulters")
        ax.plot(table["threshold"], table["cost_false_alarms"], "--", label="False alarms")
        ax.axvline(threshold, color="tab:blue", label=f"Your threshold = {threshold:.2f}")
        ax.axvline(best["threshold"], color="green", alpha=0.6, label=f"Cheapest = {best['threshold']:.2f}")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Cost (assumed units)")
        ax.legend()
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(
            [[now["tn"], now["fp"]], [now["fn"], now["tp"]]],
            index=["Actually paid", "Actually defaulted"],
            columns=["Predicted: approve", "Predicted: flag"]))
        st.write(f"**Cheapest threshold:** {best['threshold']:.2f} -> catches "
                 f"{best['defaulters_caught_recall']:.1%} of defaulters, flags {best['flagged_share']:.1%} "
                 f"of customers, total cost {best['total_cost']:,.0f}.")
        st.write(f"**Approve everyone:** cost {approve_all['total_cost']:,.0f}.")
        st.write(f"**Current project rule (high-risk cut-off {config.HIGH_RISK_MIN}):** cost "
         f"{evaluate_threshold(y, proba, config.HIGH_RISK_MIN, cost_default, cost_alarm)['total_cost']:,.0f}.")
   

with tab2:
    left, right = st.columns([3, 2])

    with left:
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.hist(proba[y == 0], bins=40, alpha=0.6, label="Actually paid")
        ax.hist(proba[y == 1], bins=40, alpha=0.6, label="Actually defaulted")
        ax.axvline(threshold, color="black", linestyle="--", label="Your threshold")

        ax.set_xlabel("Predicted default probability", fontsize=9)
        ax.set_ylabel("Number of customers", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(fontsize=8)

        fig.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        bands = pd.cut(
            proba,
            [0, config.LOW_RISK_MAX, config.HIGH_RISK_MIN, 1.0001],
            right=False,
            labels=["Low", "Medium", "High"]
        )

        summary = (
            pd.DataFrame({"band": bands, "defaulted": y})
            .groupby("band", observed=True)["defaulted"]
            .agg(customers="count", actual_default_rate="mean")
        )

        summary["actual_default_rate"] = summary["actual_default_rate"].map("{:.1%}".format)

        st.subheader("Low / Medium / High risk bands")
        st.dataframe(summary, use_container_width=True)



with tab3:
    idx = st.number_input("Test customer number (0 - 5999)", 0, len(proba) - 1, 0)
    p = float(proba[idx])
    verdict = "FLAG (predicted to default)" if p >= threshold else "APPROVE"
    st.metric("Predicted default probability", f"{p:.1%}")
    st.write(f"At your threshold of {threshold:.2f}: **{verdict}**.  "
             f"Existing band: **{config.risk_band(p)}**.  "
             f"What actually happened: **{'defaulted' if y[idx] else 'paid'}**.")
