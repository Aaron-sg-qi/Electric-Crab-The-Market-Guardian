import asyncio
import json
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

from electric_crab_core import ElectricCrabAgent, AuditResult


st.set_page_config(
    page_title="Electric Crab – The Market Guardian",
    page_icon="🦀",
    layout="wide"
)


# =========================================================
# Helpers
# =========================================================

def run_async(coro):
    """
    Streamlit-friendly async runner.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def risk_badge_color(risk_level: str) -> str:
    if risk_level == "HIGH":
        return "#ff4b4b"
    if risk_level == "MEDIUM":
        return "#ffa500"
    return "#2ecc71"


def render_risk_badge(risk_level: str):
    color = risk_badge_color(risk_level)
    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:8px 14px;
            border-radius:10px;
            color:white;
            font-weight:700;
            display:inline-block;
        ">
            {risk_level} RISK
        </div>
        """,
        unsafe_allow_html=True
    )


def audit_results_to_table(results: List[AuditResult]) -> pd.DataFrame:
    rows = []

    for result in results:
        rows.append({
            "Market": result.title,
            "Market Probability": round(result.market_probability * 100, 2),
            "Electric Crab Probability": round(result.model_probability * 100, 2),
            "Deviation": round(result.deviation * 100, 2),
            "Risk": result.risk_level,
            "Trust Score": result.trust_score,
            "Factors": ", ".join(result.main_factors[:3])
        })

    return pd.DataFrame(rows)


def render_agent_votes(result: AuditResult):
    multi_agent = result.xapi["result"]["multi_agent"]
    votes = multi_agent["agent_votes"]

    vote_rows = []

    for vote in votes:
        vote_rows.append({
            "Agent": vote["agent"],
            "Probability": round(vote["probability"] * 100, 2),
            "Confidence": vote["confidence"],
            "Signal": vote["signal"],
            "Explanation": vote["explanation"]
        })

    st.dataframe(
        pd.DataFrame(vote_rows),
        use_container_width=True,
        hide_index=True
    )


def render_xapi_tasks(result: AuditResult):
    tasks = result.xapi["result"].get("xapi_gateway_tasks", [])

    if not tasks:
        st.info("No xapi.to tasks generated for this market.")
        return

    for index, task in enumerate(tasks, start=1):
        with st.expander(f"xapi.to Task {index}: {task.get('capability_intent')}"):
            st.code(task.get("xapi_prompt", ""), language="text")
            st.json(task)


def render_market_card(result: AuditResult):
    xapi_result = result.xapi["result"]
    decision = xapi_result["decision_showcase"]
    multi_agent = xapi_result["multi_agent"]

    st.markdown("---")
    st.subheader(result.title)

    left, mid, right = st.columns([1, 1, 1])

    with left:
        st.metric(
            "Market Probability",
            f"{round(result.market_probability * 100, 2)}%"
        )

    with mid:
        st.metric(
            "Electric Crab Probability",
            f"{round(result.model_probability * 100, 2)}%",
            delta=f"{round((result.model_probability - result.market_probability) * 100, 2)}%"
        )

    with right:
        st.metric(
            "Trust Score",
            result.trust_score
        )

    badge_col, consensus_col, deviation_col = st.columns([1, 1, 1])

    with badge_col:
        render_risk_badge(result.risk_level)

    with consensus_col:
        st.write("**Consensus Level**")
        st.write(multi_agent["consensus_level"])

    with deviation_col:
        st.write("**Agent Disagreement**")
        st.write(multi_agent["agent_disagreement"])

    st.write("**TL;DR**")
    st.info(result.tldr)

    st.write("**Decision Signal**")
    st.success(f"{decision['decision_signal']} — {decision['action_label']}")

    st.write("**Main Risk Factors**")
    st.write(", ".join(result.main_factors))

    tab1, tab2, tab3, tab4 = st.tabs([
        "Agent Votes",
        "xapi.to Tasks",
        "Raw xAPI Output",
        "Decision Showcase"
    ])

    with tab1:
        render_agent_votes(result)

    with tab2:
        render_xapi_tasks(result)

    with tab3:
        st.json(result.xapi)

    with tab4:
        st.json(decision)


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("🦀 Electric Crab")
st.sidebar.write("Multi-Agent Prediction Market Intelligence")

mode = st.sidebar.radio(
    "Data Source",
    ["Demo Data", "Real Polymarket Data"]
)

use_deep_learning = st.sidebar.checkbox("Use Deep Learning", value=True)
use_reinforcement_learning = st.sidebar.checkbox("Use RL Risk Optimizer", value=True)
use_gpu = st.sidebar.checkbox("Use GPU / Batch Scoring", value=True)

notify = st.sidebar.checkbox("Generate @Mention All xapi.to Task", value=True)
send = st.sidebar.checkbox("Try Real xapi.to Send", value=False)

if send:
    st.sidebar.warning(
        "Send mode will try to execute xapi.to CLI. "
        "Use only after XAPI_API_KEY and XAPI_ENABLE_CLI are configured."
    )

limit = st.sidebar.slider(
    "Number of Real Markets",
    min_value=1,
    max_value=10,
    value=5
)


# =========================================================
# Header
# =========================================================

st.title("🦀 Electric Crab – The Market Guardian")

st.markdown(
    """
    Electric Crab is a **multi-agent AI system** for prediction market risk intelligence.

    It audits Polymarket-style markets using specialist agents for price, liquidity,
    whale risk, volatility, sentiment, classical ML, deep learning, and GPU batch scoring.
    It also prepares **xapi.to gateway tasks** for external research and `@Mention All` audit notifications.
    """
)


# =========================================================
# Main Action
# =========================================================

run_button = st.button("Run Electric Crab Audit", type="primary")

if run_button:
    with st.spinner("Electric Crab agents are auditing markets..."):
        agent = ElectricCrabAgent(
            use_real_data=(mode == "Real Polymarket Data"),
            use_deep_learning=use_deep_learning,
            use_reinforcement_learning=use_reinforcement_learning,
            use_gpu=use_gpu
        )

        try:
            if mode == "Real Polymarket Data":
                results = run_async(
                    agent.audit_real_polymarket_events(limit=limit)
                )
            else:
                demo_events = [
                    {
                        "event_id": "demo-001",
                        "title": "Will Candidate X win the election?"
                    },
                    {
                        "event_id": "demo-002",
                        "title": "Will BTC close above $100K this month?"
                    },
                    {
                        "event_id": "demo-003",
                        "title": "Will Team A win the final?"
                    },
                    {
                        "event_id": "demo-004",
                        "title": "Will a major AI regulation bill pass this year?"
                    },
                    {
                        "event_id": "demo-005",
                        "title": "Will ETH outperform BTC this quarter?"
                    }
                ]

                results = run_async(
                    agent.audit_demo_events(demo_events)
                )

            st.session_state["audit_results"] = results
            st.session_state["agent"] = agent

        except Exception as exc:
            st.error(f"Audit failed: {exc}")
            st.stop()


# =========================================================
# Results
# =========================================================

if "audit_results" in st.session_state:
    results = st.session_state["audit_results"]
    agent = st.session_state["agent"]

    st.success(f"Audit completed. {len(results)} markets analyzed.")

    summary_df = audit_results_to_table(results)

    st.subheader("Market Audit Summary")
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    high_count = sum(1 for r in results if r.risk_level == "HIGH")
    medium_count = sum(1 for r in results if r.risk_level == "MEDIUM")
    low_count = sum(1 for r in results if r.risk_level == "LOW")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Markets Audited", len(results))

    with c2:
        st.metric("High Risk", high_count)

    with c3:
        st.metric("Medium Risk", medium_count)

    with c4:
        st.metric("Low Risk", low_count)

    st.subheader("Detailed Market Cards")

    for result in results:
        render_market_card(result)

    if notify:
        st.subheader("xapi.to @Mention All Notification")

        with st.spinner("Building xapi.to @Mention All task..."):
            notify_result = run_async(
                agent.call_xapi_mention_all(
                    results=results,
                    dry_run=not send
                )
            )

        if notify_result.get("executed"):
            st.success("xapi.to task executed successfully.")
        else:
            st.warning("xapi.to task generated but not executed.")

        if notify_result.get("reason"):
            st.info(notify_result.get("reason"))

        if notify_result.get("error"):
            st.error(notify_result.get("error"))

        st.write("**Task Preview**")
        st.json(notify_result.get("task"))

        task = notify_result.get("task") or {}
        message = task.get("message")

        if message:
            st.write("**@Mention All Message**")
            st.code(message, language="text")

    st.subheader("Export JSON")

    export_payload = [
        {
            "event_id": result.event_id,
            "title": result.title,
            "market_probability": result.market_probability,
            "model_probability": result.model_probability,
            "deviation": result.deviation,
            "risk_level": result.risk_level,
            "trust_score": result.trust_score,
            "main_factors": result.main_factors,
            "tldr": result.tldr,
            "xapi": result.xapi
        }
        for result in results
    ]

    st.download_button(
        label="Download Audit Report JSON",
        data=json.dumps(export_payload, indent=2, ensure_ascii=False),
        file_name="electric_crab_audit_report.json",
        mime="application/json"
    )

else:
    st.info("Choose settings from the sidebar, then click **Run Electric Crab Audit**.")