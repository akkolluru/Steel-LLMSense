import requests
import json
import os

def build_reflection_prompt(summary, reasoning):
    objective = (
        "You are an expert system evaluator. Evaluate the LLM output analyzing steel plant data "
        "for correctness, clarity, and practical usefulness."
    )

    format_instructions = (
        "Reply strictly in JSON format:\n"
        "{\n"
        "  \"correctness_score\": \"0-10\",\n"
        "  \"clarity_score\": \"0-10\",\n"
        "  \"practical_usefulness_score\": \"0-10\",\n"
        "  \"improvement_suggestions\": \"Your concise suggestions here.\"\n"
        "}"
    )

    prompt = (
        f"OBJECTIVE:\n{objective}\n\n"
        f"SUMMARY:\n{summary}\n\n"
        f"REASONING:\n{reasoning}\n\n"
        f"FORMAT:\n{format_instructions}"
    )
    return prompt

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

def main():
    print("Reflection Evaluation using Mistral LLM")
    output_folder = "outputs/"
    files = [f for f in os.listdir(output_folder) if f.endswith('.json')]
    if not files:
        print("No JSON files found in the outputs folder.")
        return
    
    print("Available JSON files:")
    for file in files:
        print(f"- {file}")
    
    file_name_input = input("Enter the name wihtout extension")

    file_name = f"{file_name_input}.json"

    with open(os.path.join(output_folder, file_name), 'r') as f:
        data = json.load(f)
        summary = data.get("summary", "")
        reasoning = data.get("reasoning", "")

    prompt = build_reflection_prompt(summary, reasoning)
    reflection_output = send_to_ollama(prompt)
    if reflection_output:
        print(f"Ollama Response: \n{reflection_output}")
        print(reflection_output)
    
if __name__ == "__main__":
    main()