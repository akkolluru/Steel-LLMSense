import pandas as pd
import json
import os
from datetime import date, datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import requests

def load_steel_data(file_path="data\Steel_industry_data.csv"):
    """Loads and preprocesses the steel industry dataset."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")
    return df

def generate_regression_equation(df_period: pd.DataFrame) -> str:
    """
    Generates a linear regression equation from the provided data period.
    This equation serves as a dynamic summary of the data patterns.
    """
    if df_period.empty or len(df_period) < 2:
        return "Not enough data to generate a model for this period."

    dependent_var = 'Usage_kWh'
    independent_vars = [
        'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor',
        'NSM', 'WeekStatus', 'Day_of_week', 'Load_Type'
    ]

    X = df_period[independent_vars]
    y = df_period[dependent_var]

    categorical_features = ['WeekStatus', 'Day_of_week', 'Load_Type']
    numerical_features = [col for col in independent_vars if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X, y)

    intercept = pipeline.named_steps['regressor'].intercept_
    coefficients = pipeline.named_steps['regressor'].coef_
    feature_names = (numerical_features +
                   list(pipeline.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .get_feature_names_out(categorical_features)))

    equation = f"{dependent_var} = {intercept:.4f}"
    for coef, name in zip(coefficients, feature_names):
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {abs(coef):.4f} * {name}"

    return equation

def summarize_day(df: pd.DataFrame, day: date) -> str:
    """Summarizes a single day by generating a regression equation."""
    day_dt = pd.to_datetime(day)
    df_day = df[df['date'].dt.date == day_dt.date()]
    return generate_regression_equation(df_day)

def summarize_period(df: pd.DataFrame, start: date, end: date) -> str:
    """Summarizes a date range by generating a regression equation."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    df_period = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
    return generate_regression_equation(df_period)

def build_prompt(equation: str) -> str:
    """Builds the prompt for the LLM to interpret the equation."""
    prompt = f"""
    You are an expert industrial data analyst. Your task is to interpret a regression model and provide a simple, actionable summary for a plant manager.

    Here is the regression equation summarizing the plant's recent performance:
    ```
    {equation}
    ```

    Based on this model, please provide the following in a strict JSON format:

    1.  **"maintenance_needed"**: A boolean (true/false). Set to `true` if the model indicates any anomalies or inefficiencies (e.g., high impact from lagging reactive power, poor power factor correlation) that suggest equipment needs inspection.
    2.  **"reasoning"**: A concise, layman's terms explanation of the key factors driving energy consumption (`Usage_kWh`) according to the model. Explain what the most impactful variables are.
    3.  **"energy_optimization_suggestions"**: Provide 2-3 bullet-pointed, practical suggestions for how the plant can optimize its energy usage based on the relationships in the equation.

    Example Output Format:
    {{
      "maintenance_needed": true,
      "reasoning": "Energy consumption is heavily driven by reactive power, indicating potential inefficiencies in motors or transformers. The type of load also has a significant impact, with heavy loads consuming proportionally more power.",
      "energy_optimization_suggestions": [
        "Investigate equipment with high lagging reactive power to improve power factor.",
        "Shift non-essential heavy loads to off-peak hours if possible.",
        "Review the efficiency of machinery used during 'Maximum_Load' periods."
      ]
    }}
    """
    return prompt

def send_to_ollama(prompt_text: str, model: str = "mistral", temperature: float = 0.1) -> str:
    """Sends the prompt to the Ollama API and returns the response."""
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
        return f'{{"error": "Failed to connect to Ollama API: {e}"}}'

def save_output_json(summary_equation: str, llm_reasoning: str, filename: str):
    """Saves the summary and LLM reasoning to a JSON file."""
    output_folder = "outputs/"
    os.makedirs(output_folder, exist_ok=True)

    try:
        # The LLM reasoning is already expected to be a JSON string
        reasoning_json = json.loads(llm_reasoning)
    except json.JSONDecodeError:
        reasoning_json = {"raw_output": llm_reasoning}

    output_data = {
        "summary_equation": summary_equation,
        "reasoning": reasoning_json
    }
    file_path = os.path.join(output_folder, f"{filename}.json")
    with open(file_path, 'w') as f:
        json.dump(output_data, f, indent=4)