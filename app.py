import streamlit as st
import os
import json
import pandas as pd
from datetime import date

# Import summary + LLM tools
from scripts.generate_daily_summaries import (
    load_steel_data,
    summarize_day,
    summarize_period,
    summarize_all_days,
    build_prompt,
    send_to_ollama,
    save_output_json
)

from scripts.evaluate_llm import (
    build_reflection_prompt,
    build_chain_of_thought_prompt,
    build_single_debate_prompt,
    run_iterative_debate,
    send_to_ollama as eval_send_to_ollama,
    save_revised_reasoning
)

OUTPUT_FOLDER = "outputs/"
df = load_steel_data()

st.set_page_config(page_title="LLMSense Evaluation App", layout="centered")
st.title("LLMSense Evaluation Tool")

st.sidebar.header("Generate Summary")
summary_mode = st.sidebar.radio("Summarize by:", ["Single Day", "Date Range", "Month"])

if summary_mode == "Single Day":
    selected_day = st.sidebar.date_input("Select date", value=df['date'].min().date())
    summary = summarize_day(df, selected_day)

elif summary_mode == "Date Range":
    start = st.sidebar.date_input("Start date", value=df['date'].min().date())
    end = st.sidebar.date_input("End date", value=df['date'].max().date())
    summary = summarize_period(df, start, end)

elif summary_mode == "Month":
    month_list = pd.to_datetime(df['date']).dt.to_period("M").astype(str).unique()
    selected_month = st.sidebar.selectbox("Choose month", month_list)
    start = pd.to_datetime(f"{selected_month}-01")
    end = (start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
    summary = summarize_period(df, start, end)

st.subheader("Generated Summary")
st.code(summary)

if st.button("Generate LLM Reasoning"):
    with st.spinner("Sending summary to LLM..."):
        prompt = build_prompt(summary)
        response = send_to_ollama(prompt)

        # Save state for saving later
        st.session_state["latest_summary"] = summary
        st.session_state["latest_reasoning"] = response

        st.success("LLM Reasoning generated.")

if "latest_reasoning" in st.session_state:
    st.subheader("LLM Reasoning Output")

    try:
        parsed = json.loads(st.session_state["latest_reasoning"])
        st.markdown(f"""
**Maintenance Needed**: `{parsed.get("maintenance_needed", "N/A")}`  
**Reasoning**: {parsed.get("reasoning", "N/A")}  
**Energy Optimization Suggestions**: {parsed.get("energy_optimization_suggestions", "N/A")}
        """)
    except:
        st.code(st.session_state["latest_reasoning"])

    if st.checkbox("Save summary and reasoning"):
        filename = st.text_input("File name (without .json):")
        if st.button("Save File"):
            if filename:
                save_output_json(
                    st.session_state["latest_summary"],
                    st.session_state["latest_reasoning"],
                    filename
                )
                st.success(f"File saved: outputs/{filename}.json")
            else:
                st.warning("Please enter a valid filename.")

# Divider
st.markdown("---")

# Evaluation Section
st.header("Evaluate Existing Summaries")
json_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.json') and not f.endswith('_revised.json')]
selected_file = st.selectbox("Choose a file to evaluate", json_files)

if selected_file:
    with open(os.path.join(OUTPUT_FOLDER, selected_file)) as f:
        data = json.load(f)
        summary = data.get("summary", "")
        reasoning = data.get("reasoning", "")

    st.subheader("Summary")
    st.code(summary)

    st.subheader("Reasoning")
    try:
        parsed = json.loads(reasoning)
        st.markdown(f"""
**Maintenance Needed**: `{parsed.get("maintenance_needed", "N/A")}`  
**Reasoning**: {parsed.get("reasoning", "N/A")}  
**Energy Optimization Suggestions**: {parsed.get("energy_optimization_suggestions", "N/A")}
        """)
    except:
        st.code(reasoning)

    eval_method = st.radio("Evaluation Method", [
        "Reflection",
        "Chain of Thought",
        "Single-Round Debate",
        "Iterative Agent Debate"
    ])

    if st.button("Run Evaluation"):
        with st.spinner("Evaluating with model..."):
            if eval_method == "Reflection":
                result = eval_send_to_ollama(build_reflection_prompt(summary, reasoning))
            elif eval_method == "Chain of Thought":
                result = eval_send_to_ollama(build_chain_of_thought_prompt(summary, reasoning))
            elif eval_method == "Single-Round Debate":
                result = eval_send_to_ollama(build_single_debate_prompt(summary, reasoning))
            elif eval_method == "Iterative Agent Debate":
                result = run_iterative_debate(summary, reasoning)
                revised = result.get("revised_pro", "")
                save_revised_reasoning(selected_file.replace(".json", ""), summary, revised)

        st.subheader("Evaluation Result")

        if isinstance(result, dict):
            for key, val in result.items():
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                st.code(val.strip() if isinstance(val, str) else json.dumps(val, indent=2))
        else:
            st.code(result.strip())
