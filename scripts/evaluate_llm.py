import requests
import json
import os

def send_to_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    response = requests.post(
        url,
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    if response.status_code != 200:
        print("Error communicating with Ollama:", response.text)
        return None
    return response.json()['response']

# ------------------ PROMPT BUILDERS ------------------

def build_reflection_prompt(summary, reasoning):
    return (
        "You are an expert system evaluator. Evaluate the LLM output analyzing steel plant data "
        "for correctness, clarity, and practical usefulness.\n\n"
        f"SUMMARY:\n{summary}\n\n"
        f"REASONING:\n{reasoning}\n\n"
        "Reply strictly in JSON format:\n"
        "{\n"
        "  \"correctness_score\": \"0-10\",\n"
        "  \"clarity_score\": \"0-10\",\n"
        "  \"practical_usefulness_score\": \"0-10\",\n"
        "  \"improvement_suggestions\": \"Your concise suggestions here.\"\n"
        "}"
    )

def build_chain_of_thought_prompt(summary, reasoning):
    return (
        f"You are a senior AI evaluator. Carefully analyze the following step-by-step:\n\n"
        f"STEP 1: Understand the summary.\n{summary}\n\n"
        f"STEP 2: Evaluate reasoning.\n{reasoning}\n\n"
        f"STEP 3: Think step-by-step: Is the analysis accurate, clear, and useful?\n\n"
        "Reply in JSON format:\n"
        "{\n"
        "  \"correctness\": \"0-10\",\n"
        "  \"clarity\": \"0-10\",\n"
        "  \"usefulness\": \"0-10\",\n"
        "  \"thoughts\": \"Step-by-step explanation\"\n"
        "}"
    )

def build_single_debate_prompt(summary, reasoning):
    return (
        f"SUMMARY:\n{summary}\n\n"
        f"REASONING:\n{reasoning}\n\n"
        "Agent Pro: Explain why the reasoning is solid and insightful.\n"
        "Agent Con: Explain potential flaws or unclear points.\n"
        "Judge: Consider both arguments and assign scores.\n\n"
        "Reply in JSON:\n"
        "{\n"
        "  \"pro_argument\": \"...\",\n"
        "  \"con_argument\": \"...\",\n"
        "  \"judge_summary\": \"...\",\n"
        "  \"scores\": { \"correctness\": \"0-10\", \"clarity\": \"0-10\", \"usefulness\": \"0-10\" }\n"
        "}"
    )

def build_pro_agent_prompt(summary, reasoning):
    return (
        "You are Agent Pro. Defend the reasoning below as useful, correct, and clear.\n\n"
        f"SUMMARY:\n{summary}\n\nREASONING:\n{reasoning}\n\n"
        "Give a strong argument in support."
    )

def build_con_agent_prompt(summary, pro_argument):
    return (
        "You are Agent Con, a sharp and honest evaluator.\n"
        "Your job is to strictly critique Agent Pro’s reasoning ONLY if there are real issues.\n"
        "Avoid generic or vague complaints like 'not enough data' or 'lack of context' unless clearly justified by the summary.\n\n"
        "If the reasoning is logically sound, practical, and clearly expressed, then state:\n"
        "'The reasoning is strong. No major issues to critique.'\n\n"
        f"SUMMARY:\n{summary}\n\n"
        f"AGENT PRO'S ARGUMENT:\n{pro_argument}\n\n"
        "Provide only solid critiques. If there are no issues, say so directly."
    )


def build_pro_revision_prompt(summary, con_argument):
    return (
        "You are Agent Pro again. Revise your original reasoning to address Agent Con's criticism.\n\n"
        f"SUMMARY:\n{summary}\n\n"
        f"CON ARGUMENT:\n{con_argument}\n\n"
        "Reply with a stronger and improved version of your reasoning."
    )

def build_judge_prompt(summary, pro_argument, con_argument, revised_pro):
    return (
        "You are the Judge. Consider the full debate below and evaluate the final revised reasoning.\n\n"
        f"SUMMARY:\n{summary}\n\n"
        f"Agent Pro (Initial):\n{pro_argument}\n\n"
        f"Agent Con:\n{con_argument}\n\n"
        f"Agent Pro (Revised):\n{revised_pro}\n\n"
        "Now score the final reasoning in JSON format:\n"
        "{\n"
        "  \"final_scores\": { \"correctness\": \"0-10\", \"clarity\": \"0-10\", \"usefulness\": \"0-10\" },\n"
        "  \"final_comments\": \"Judge’s evaluation of the improved answer.\"\n"
        "}"
    )

# ------------------ MAIN LOGIC ------------------

def run_iterative_debate(summary, reasoning):
    print("➤ Round 1: Agent Pro")
    pro_arg = send_to_ollama(build_pro_agent_prompt(summary, reasoning))
    print(pro_arg)

    print("\n➤ Round 2: Agent Con")
    con_arg = send_to_ollama(build_con_agent_prompt(summary, pro_arg))
    print(con_arg)

    print("\n➤ Round 3: Agent Pro Revises")
    revised_pro = send_to_ollama(build_pro_revision_prompt(summary, con_arg))
    print(revised_pro)

    print("\n➤ Final Judgement")
    judge = send_to_ollama(build_judge_prompt(summary, pro_arg, con_arg, revised_pro))
    print(judge)

    return {
        "pro_argument": pro_arg,
        "con_argument": con_arg,
        "revised_pro": revised_pro,
        "judge": judge
    }
def save_revised_reasoning(original_filename, summary, revised_reasoning):
    revised_data = {
        "summary": summary,
        "revised_reasoning": revised_reasoning
    }

    new_filename = f"{original_filename}_revised.json"
    output_path = os.path.join("outputs", new_filename)
    
    with open(output_path, 'w') as f:
        json.dump(revised_data, f, indent=4)
    
    print(f"\n✅ Revised reasoning saved to: outputs/{new_filename}")

def main():
    print("LLM Evaluation System (Reflection / CoT / Debate / Iterative Debate)")
    output_folder = "outputs/"
    files = [f for f in os.listdir(output_folder) if f.endswith('.json')]
    if not files:
        print("No JSON files found in the outputs folder.")
        return

    print("\nAvailable JSON files:")
    for file in files:
        print(f"- {file}")

    file_name_input = input("\nEnter the file name (without .json): ").strip()
    file_path = os.path.join(output_folder, f"{file_name_input}.json")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            summary = data.get("summary", "")
            reasoning = data.get("reasoning", "")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("\nChoose evaluation method:")
    print("1. Reflection")
    print("2. Chain of Thought (CoT)")
    print("3. Single-Round Agent Debate")
    print("4. Iterative Multi-Turn Debate")
    method = input("Enter choice (1 / 2 / 3 / 4): ").strip()

    if method == '1':
        prompt = build_reflection_prompt(summary, reasoning)
        result = send_to_ollama(prompt)
    elif method == '2':
        prompt = build_chain_of_thought_prompt(summary, reasoning)
        result = send_to_ollama(prompt)
    elif method == '3':
        prompt = build_single_debate_prompt(summary, reasoning)
        result = send_to_ollama(prompt)
    elif method == '4':
        result = run_iterative_debate(summary, reasoning)
        
        # Extract and save revised reasoning
        revised_reasoning = result.get("revised_pro", "").strip()
        save_revised_reasoning(file_name_input, summary, revised_reasoning)

    else:
        print("Invalid method.")
        return

    print("\n====== Final Evaluation Output ======")
    print(json.dumps(result, indent=2) if isinstance(result, dict) else result)

if __name__ == "__main__":
    main()
