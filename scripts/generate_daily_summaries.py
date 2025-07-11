import pandas as pd
import json
import os

def load_steel_data(path="data/Steel_industry_data.csv"):
    
    #Load steel industry data from a CSV file.
    
    df =  pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")

    return df

def summarize_day(df, date):
    day_df = df[df['date'].dt.date == pd.to_datetime(date).date()]
    if day_df.empty:
        return f"No data found for {date}."
    
    summary = f"Summary for {date}:\n"

    numeric_columns = [
        'Usage_kWh',
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor'
    ]

    for col in numeric_columns:
        mean_val = day_df[col].mean()
        max_val = day_df[col].max()
        min_val = day_df[col].min()
        summary += (
            f"- {col}: Avg = {mean_val:.2f}, Max = {max_val:.2f}, Min = {min_val:.2f}\n"
        )

    load_type_counts = day_df['Load_Type'].value_counts(normalize=True) * 100
    summary += "\nLoad Type Distribution:\n"

    for load_type, percentage in load_type_counts.items():
        summary += f"- {load_type}: {percentage:.2f}%\n"

    return summary

def summarize_all_days(df):
    all_summaries = {}
    unique_dates = df['date'].dt.date.unique()
    for date in unique_dates:
        summary = summarize_day(df, date)
        all_summaries[str(date)] = summary
    return all_summaries

def summarize_period(df, start_date, end_date, label="period"):
    period_df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    if period_df.empty:
        return f"No data found for the period from {start_date} to {end_date}."
    summary = f"Summary from {start_date} to {end_date}:\n"
    numeric_columns = [
        'Usage_kWh',
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor'
    ]
    for col in numeric_columns:
        summary += (
            f"- {col}: Avg = {period_df[col].mean():.2f}, "
            f"Max = {period_df[col].max():.2f}, "
            f"Min = {period_df[col].min():.2f}\n"
        )
    load_type_counts = period_df['Load_Type'].value_counts(normalize=True) * 100
    summary += "\nLoad Type Distribution:\n"
    for load_type, percentage in load_type_counts.items():
        summary += f"- {load_type}: {percentage:.2f}%\n"
    return summary

import requests
def build_prompt(summary):
    objective = (
        "You are an expert industrial process analyst. "
        "Your task is to analyze steel plant daily sensor summaries to detect anomalies, "
        "predict potential maintenance needs, and suggest energy optimization."
    )
    context = (
        "The steel plant data includes:\n"
        "- Usage_kWh: Energy usage\n"
        "- Lagging_Current_Reactive.Power_kVarh: Reactive power\n"
        "- Leading_Current_Reactive_Power_kVarh: Reactive power\n"
        "- CO2(tCO2): Emissions\n"
        "- Power Factors: Operational efficiency\n"
        "- Load Type distribution: Indicates operational states\n"
        "Analyze based on your domain knowledge."
    )
    format_instructions = (
        "Reply strictly in the following JSON format:\n"
        "{\n"
        "  \"maintenance_needed\": \"Yes/No\",\n"
        "  \"reasoning\": \"Your concise reasoning here.\",\n"
        "  \"energy_optimization_suggestions\": \"Your suggestions here.\"\n"
        "}"
    )
    prompt = (
        f"OBJECTIVE:\n{objective}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"DATA:\n{summary}\n\n"
        f"FORMAT:\n{format_instructions}"
    )
    return prompt

def send_to_ollama(prompt):
    url="http://localhost:11434/api/generate"
    response = requests.post(
        url,
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    if response.status_code != 200:
        print("Error communicating with Ollama:", response.text)
        return None
    return response.json()['response']

def save_output_json(summary, reasoning, suggested_name):
    output_data = {
        "summary": summary,
        "reasoning": reasoning
    }

    file_path = f"outputs/{suggested_name}.json"
    with open(file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Output saved to {file_path}")

if __name__ == "__main__":
    df = load_steel_data()

    print("Steel LLMSense Summary Generator")
    print("1. Summarize all days")
    print("2. Summarize a specific day")
    print("3. Summarize a period")
    print("4. Summarize a mothn")
    choice = input("Enter your choice (1, 2, 3, or 4): ")

    if choice == '1':
        summaries = summarize_all_days(df)
        for date, summary in summaries.items():
            print(f"\n{date} Summary:\n{summary}")
            prompt = build_prompt(summary)
            response = send_to_ollama(prompt)
            if response:
                print(f"Ollama Response for {date}:\n{response}")
            print("-"* 40)
            print("would you like to save the output? (yes/no)")
            save_choice = input().strip().lower()
            if save_choice == 'yes':
                suggested_name = input("Enter a name for the output file (without extension): ").strip()
                save_output_json(summary, response, suggested_name)
    elif choice == '2':
        date = input("Enter the date (YYYY-MM-DD): ")
        summary = summarize_day(df, date)
        print(summary)
        prompt = build_prompt(summary)
        response = send_to_ollama(prompt)
        if response:
            print(f"Ollama Response for {date}:\n{response}")
        print("would you like to save the output? (yes/no)")
        save_choice = input().strip().lower()
        if save_choice == 'yes':
            suggested_name = input("Enter a name for the output file (without extension): ").strip()
            save_output_json(summary, response, suggested_name)
    elif choice == '3':
        start_date = input("Enter the start date (YYYY-MM-DD): ")
        end_date = input("Enter the end date (YYYY-MM-DD): ")
        summary = summarize_period(df, start_date, end_date)
        print(summary)
        prompt = build_prompt(summary)
        response = send_to_ollama(prompt)
        if response:
            print(f"Ollama Response for period {start_date} to {end_date}:\n{response}")
        print("would you like to save the output? (yes/no)")
        save_choice = input().strip().lower()
        if save_choice == 'yes':
            suggested_name = input("Enter a name for the output file (without extension): ").strip()
            save_output_json(summary, response, suggested_name)
    elif choice == '4':
        month = input("Enter the month (YYYY-MM): ")
        start_date = pd.to_datetime(month + "-01")
        next_month = start_date + pd.DateOffset(months=1)
        end_date = next_month.strftime("%Y-%m-%d")
        start_date = start_date.strftime("%Y-%m-%d")
        summary = summarize_period(df, start_date, end_date, label="month")
        print(summary)
        prompt = build_prompt(summary)
        response = send_to_ollama(prompt)
        if response:
            print(f"Ollama Response for month {month}:\n{response}")
        print("would you like to save the output? (yes/no)")
        save_choice = input().strip().lower()
        if save_choice == 'yes':
            suggested_name = input("Enter a name for the output file (without extension): ").strip()
            save_output_json(summary, response, suggested_name)
    
    else:
        print("Invalid choice. Please enter 1 to 4.")
    
    
    
