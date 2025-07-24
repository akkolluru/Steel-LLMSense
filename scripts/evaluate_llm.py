import json
import requests
import os
from tenacity import retry, stop_after_attempt, wait_fixed

# --------------------------------------------------------------------------
# OLLAMA API COMMUNICATION
# --------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def send_to_ollama(prompt_text: str, model: str = "mistral", temperature: float = 0.1) -> str:
    """
    Sends a prompt to the Ollama API and returns the response.
    Includes retry logic for network robustness.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        # This will be returned after the final retry attempt fails
        return f'{{"error": "Failed to connect to Ollama API after multiple attempts: {e}"}}'

# --------------------------------------------------------------------------
# METHOD 1: STANDARD REFLECTION
# --------------------------------------------------------------------------

def build_reflection_prompt(summary_equation: str, llm_reasoning: str) -> str:
    """
    Builds a standard reflection prompt to evaluate the LLM's interpretation.
    """
    return f"""
    You are an expert system designed to evaluate the quality of LLM-generated analysis.

    **Original Data Summary (The Model Equation):**
    ```
    {summary_equation}
    ```

    **LLM's Interpretation and Reasoning:**
    ```
    {llm_reasoning}
    ```

    Please evaluate the LLM's reasoning based on the equation. Provide your output in a strict JSON format with the following keys:
    - "correctness_score" (0-10): How accurately does the LLM interpret the key drivers from the equation?
    - "clarity_score" (0-10): How clear and easy to understand is the layman's explanation?
    - "practical_usefulness_score" (0-10): How actionable and relevant are the optimization suggestions?
    - "improvement_suggestions": A brief summary of what the LLM could do to improve its analysis.
    """

# --------------------------------------------------------------------------
# METHOD 2: CHAIN OF THOUGHT (CoT) REFLECTION
# --------------------------------------------------------------------------

def build_chain_of_thought_prompt(summary_equation: str, llm_reasoning: str) -> str:
    """
    Builds a Chain of Thought prompt for a more detailed and reasoned evaluation.
    """
    return f"""
    You are an expert system designed to evaluate the quality of LLM-generated analysis.

    **Original Data Summary (The Model Equation):**
    ```
    {summary_equation}
    ```

    **LLM's Interpretation and Reasoning:**
    ```
    {llm_reasoning}
    ```

    Let's think step by step to evaluate the LLM's output.
    1.  **Analyze the Equation**: First, identify the variables with the largest absolute coefficients in the equation. These are the most significant drivers of `Usage_kWh`. List them.
    2.  **Check for Correctness**: Compare the LLM's "reasoning" with the key drivers you identified. Did the LLM correctly spot the most impactful variables? Does the reasoning logically follow from the mathematical signs (+/-) of the coefficients?
    3.  **Assess Clarity**: Read the "reasoning" from the perspective of a non-technical plant manager. Is the language simple, direct, and free of jargon?
    4.  **Evaluate Suggestions**: Are the "energy_optimization_suggestions" directly and logically derived from the equation? For example, if `Lagging_Current_Reactive.Power_kVarh` has a high positive coefficient, a suggestion to improve the power factor is highly relevant and useful.

    After this step-by-step analysis, provide your final evaluation in a strict JSON format. The JSON should contain two main keys: "thought_process" (with your analysis from steps 1-4) and "final_scores" (with scores for correctness, clarity, and usefulness, plus improvement suggestions).
    """

# --------------------------------------------------------------------------
# METHOD 3: SINGLE-ROUND DEBATE
# --------------------------------------------------------------------------

def build_single_debate_prompt(summary_equation: str, llm_reasoning: str) -> str:
    """
    Builds a prompt for a single-round debate between two AI agents, judged by a third.
    """
    return f"""
    You are a moderator for an AI agent debate. Your task is to present the arguments from two opposing agents and then act as a judge to declare a winner.

    **Topic**: The quality of an LLM's analysis of a regression model.
    **Model Equation**: `{summary_equation}`
    **Original Analysis to be Debated**: `{llm_reasoning}`

    **Agent A (Pro Argument)**: "The analysis is excellent. It correctly identifies the main drivers of energy usage and provides clear, actionable advice that is directly supported by the model's coefficients."

    **Agent B (Con Argument)**: "The analysis is flawed. It either misinterprets the significance of the variables, overlooks critical interactions, or provides generic advice that isn't specifically tailored to the mathematical evidence in the equation."

    **Your Task as Judge**:
    1.  Analyze both the pro and con arguments in light of the equation and the original analysis.
    2.  Write a brief "Judge's Ruling" explaining which agent presented a more compelling case and why.
    3.  Provide a final JSON object with scores for the original analysis.

    **Output Format (Strict JSON):**
    {{
      "judges_ruling": "A brief text explaining your decision.",
      "winning_argument": "Agent A" or "Agent B",
      "scores": {{
        "correctness_score": 0-10,
        "clarity_score": 0-10,
        "practical_usefulness_score": 0-10
      }}
    }}
    """

# --------------------------------------------------------------------------
# METHOD 4: ITERATIVE AGENT DEBATE
# --------------------------------------------------------------------------

def _build_debate_agent_prompt(summary_equation: str, llm_reasoning: str, stance: str, history: str = "") -> str:
    """Helper to build prompts for the iterative debate agents."""
    if stance == "pro":
        return f"""You are a debate agent. Your goal is to DEFEND the following analysis based on the provided equation. Be specific and use the numbers from the equation to back up your points.

        Equation: `{summary_equation}`
        Analysis to Defend: `{llm_reasoning}`
        {history}
        Your turn. State your case concisely:"""
    else: # con
        return f"""You are a debate agent. Your goal is to CRITIQUE the following analysis based on the provided equation. Find flaws, missed insights, or unclear points. Be specific.

        Equation: `{summary_equation}`
        Analysis to Critique: `{llm_reasoning}`
        {history}
        Your turn. State your critique concisely:"""

def run_iterative_debate(summary_equation: str, llm_reasoning: str, rounds: int = 2) -> dict:
    """
    Manages a multi-round debate between two AI agents and returns the judged result.
    """
    history = "Debate History:\n"
    for i in range(rounds):
        # Pro Agent's Turn
        pro_prompt = _build_debate_agent_prompt(summary_equation, llm_reasoning, "pro", history)
        pro_argument = send_to_ollama(pro_prompt, temperature=0.5)
        history += f"Round {i+1} (Pro): {pro_argument}\n"

        # Con Agent's Turn
        con_prompt = _build_debate_agent_prompt(summary_equation, llm_reasoning, "con", history)
        con_argument = send_to_ollama(con_prompt, temperature=0.5)
        history += f"Round {i+1} (Con): {con_argument}\n"

    # Judge's Final Turn
    judge_prompt = f"""
    You are the judge of an AI debate. Below is the full transcript.
    Your task is to analyze the debate, declare a winner, and provide a revised, improved version of the original analysis that incorporates the valid points from both sides.

    **Original Model Equation**:
    `{summary_equation}`

    **Original Analysis**:
    `{llm_reasoning}`

    **Debate Transcript**:
    {history}

    **Your Final Judgement (Strict JSON Output):**
    {{
      "debate_summary": "A brief summary of the key arguments from both sides.",
      "winner": "Pro Agent" or "Con Agent",
      "reason_for_decision": "Explain your ruling.",
      "final_scores": {{"correctness_score": 0-10, "clarity_score": 0-10, "usefulness_score": 0-10}},
      "revised_pro": "Provide a new, improved version of the analysis here. This should be a complete JSON object with 'maintenance_needed', 'reasoning', and 'energy_optimization_suggestions' keys."
    }}
    """
    final_judgement_str = send_to_ollama(judge_prompt)
    try:
        return json.loads(final_judgement_str)
    except json.JSONDecodeError:
        return {"error": "Failed to parse the judge's final JSON output.", "raw_output": final_judgement_str}


# --------------------------------------------------------------------------
# UTILITY FUNCTION
# --------------------------------------------------------------------------

def save_revised_reasoning(filename_base: str, summary_equation: str, revised_reasoning: str):
    """
    Saves the improved/revised reasoning from a debate evaluation to a new file.
    """
    output_folder = "outputs/"
    os.makedirs(output_folder, exist_ok=True)

    # The revised reasoning might be a string representation of a dict
    try:
        reasoning_data = json.loads(revised_reasoning)
    except (json.JSONDecodeError, TypeError):
        reasoning_data = {"raw_revised_output": revised_reasoning}

    output_data = {
        "summary_equation": summary_equation,
        "revised_reasoning": reasoning_data
    }
    file_path = os.path.join(output_folder, f"{filename_base}_revised.json")
    with open(file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved revised analysis to {file_path}")