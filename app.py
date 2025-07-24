import streamlit as st
import os
import json
import pandas as pd
import mlflow
from datetime import date

# Import all necessary functions from your scripts
from scripts.generate_daily_summaries import (
    load_steel_data,
    summarize_day,
    summarize_period,
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

st.set_page_config(page_title="LLMSense Evaluation App", layout="wide")
st.title("💡 LLMSense: Regression-Based Insights & Full Evaluation Suite")

# --- MLflow Logging Helper (remains the same) ---
def log_streamlit_evaluation(file_name, method_name, result_dict):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("LLMSense Evaluations")

    with mlflow.start_run(run_name=f"{file_name}_{method_name}"):
        mlflow.set_tag("file_name", file_name)
        mlflow.set_tag("evaluation_method", method_name)

        scores = result_dict.get("scores", result_dict.get("final_scores", result_dict))
        
        # Normalize to float out of 10
        for key, ml_key in [
            ("correctness", "correctness"),
            ("correctness_score", "correctness"),
            ("clarity", "clarity"),
            ("clarity_score", "clarity"),
            ("usefulness", "usefulness"),
            ("practical_usefulness_score", "usefulness")
        ]:
            val = scores.get(key)
            if val:
                try:
                    mlflow.log_metric(ml_key, float(val))
                except (ValueError, TypeError):
                    pass

        os.makedirs("temp", exist_ok=True)
        # Log the full result JSON
        with open("temp/result.json", "w") as f:
            json.dump(result_dict, f, indent=4)
        mlflow.log_artifact("temp/result.json", "evaluation_output")

        # If it was a debate that produced a revised version, log it
        if "revised_pro" in result_dict:
            with open("temp/revised_reasoning.txt", "w") as f:
                f.write(result_dict["revised_pro"])
            mlflow.log_artifact("temp/revised_reasoning.txt")


# --- App Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Generate Model")
    summary_mode = st.radio("Summarize by:", ["Single Day", "Date Range", "Month"])

    if summary_mode == "Single Day":
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        selected_day = st.date_input("Select date", value=min_date, min_value=min_date, max_value=max_date)
        summary_equation = summarize_day(df, selected_day)
    elif summary_mode == "Date Range":
        start = st.date_input("Start date", value=df['date'].min().date())
        end = st.date_input("End date", value=df['date'].max().date())
        summary_equation = summarize_period(df, start, end)
    else: # Month
        month_list = pd.to_datetime(df['date']).dt.to_period("M").astype(str).unique()
        selected_month = st.selectbox("Choose month", month_list)
        start = pd.to_datetime(f"{selected_month}-01")
        end = (start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
        summary_equation = summarize_period(df, start.date(), end.date())

    st.subheader("Generated Regression Model")
    st.text_area("Equation", summary_equation, height=150)

    if st.button("Generate LLM Insights"):
        with st.spinner("Sending model to LLM for analysis..."):
            prompt = build_prompt(summary_equation)
            response = send_to_ollama(prompt)
            st.session_state["latest_summary_equation"] = summary_equation
            st.session_state["latest_reasoning"] = response
            st.success("LLM Insights Generated!")

with col2:
    st.header("2. LLM Analysis & Evaluation")
    if "latest_reasoning" in st.session_state:
        st.subheader("LLM Insights")
        try:
            parsed = json.loads(st.session_state["latest_reasoning"])
            maintenance_needed = parsed.get("maintenance_needed", False)
            if maintenance_needed:
                st.error(f"**Maintenance Required**: {maintenance_needed}")
            else:
                st.success(f"**Maintenance Required**: {maintenance_needed}")

            st.markdown("**Plain-English Reasoning**")
            st.info(parsed.get("reasoning", "N/A"))

            st.markdown("**Energy Optimization Suggestions**")
            suggestions = parsed.get("energy_optimization_suggestions", [])
            for sug in suggestions:
                st.markdown(f"- {sug}")

        except json.JSONDecodeError:
            st.code(st.session_state["latest_reasoning"])

        with st.expander("Save or Evaluate this Analysis"):
            # --- File Saving ---
            filename = st.text_input("File name for saving (no extension):", f"analysis_{date.today().strftime('%Y%m%d')}")
            if st.button("Save Analysis"):
                if filename:
                    save_output_json(
                        st.session_state["latest_summary_equation"],
                        st.session_state["latest_reasoning"],
                        filename
                    )
                    st.success(f"File saved: outputs/{filename}.json")
                else:
                    st.warning("Please enter a valid filename.")

            st.markdown("---")
            
            # --- Evaluation Section ---
            st.subheader("Run Evaluation on a Saved File")
            json_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.json') and not f.endswith('_revised.json')]
            if not json_files:
                st.warning("No saved analysis files found in 'outputs/' folder to evaluate.")
            else:
                selected_file = st.selectbox("Choose a file to evaluate", json_files)
                if selected_file:
                    # **THIS IS THE CORRECTED PART**
                    eval_method = st.radio("Evaluation Method", [
                        "Reflection",
                        "Chain of Thought",
                        "Single-Round Debate",
                        "Iterative Agent Debate"
                    ])

                    if st.button("Run Evaluation"):
                        with open(os.path.join(OUTPUT_FOLDER, selected_file)) as f:
                            data = json.load(f)
                            equation = data.get("summary_equation", "")
                            # Ensure reasoning is a properly formatted JSON string for the prompt
                            reasoning = json.dumps(data.get("reasoning", ""), indent=2)

                        with st.spinner(f"Evaluating with {eval_method}..."):
                            # **THIS LOGIC NOW INCLUDES ALL METHODS**
                            if eval_method == "Reflection":
                                result = eval_send_to_ollama(build_reflection_prompt(equation, reasoning))
                            elif eval_method == "Chain of Thought":
                                result = eval_send_to_ollama(build_chain_of_thought_prompt(equation, reasoning))
                            elif eval_method == "Single-Round Debate":
                                result = eval_send_to_ollama(build_single_debate_prompt(equation, reasoning))
                            elif eval_method == "Iterative Agent Debate":
                                # This function returns a dict directly
                                result = run_iterative_debate(equation, reasoning)
                                # Save the improved output from the debate
                                revised = result.get("revised_pro", "")
                                if revised:
                                    save_revised_reasoning(selected_file.replace(".json", ""), equation, revised)

                            st.subheader("Evaluation Result")
                            
                            # Display result (dict or string)
                            if isinstance(result, dict):
                                st.json(result)
                            else:
                                st.text_area("Raw Output", result, height=250)

                            # Try to parse for logging
                            try:
                                parsed_result = json.loads(result) if isinstance(result, str) else result
                            except json.JSONDecodeError:
                                parsed_result = {"raw_result": result}
                            
                            # Map to a clean tag for MLflow
                            method_tag_map = {
                                "Reflection": "Reflection",
                                "Chain of Thought": "CoT",
                                "Single-Round Debate": "Debate",
                                "Iterative Agent Debate": "IterativeDebate"
                            }
                            
                            log_streamlit_evaluation(
                                selected_file.replace(".json", ""),
                                method_tag_map.get(eval_method, "unknown"),
                                parsed_result
                            )
                            st.success(f"Evaluation using {eval_method} logged to MLflow!")