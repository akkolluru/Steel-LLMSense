import pandas as pd

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

if __name__ == "__main__":
    df = load_steel_data()

    print("Steel LLMSense Summary Generator")
    print("1. Summarize all days")
    print("2. Summarize a specific day")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        summaries = summarize_all_days(df)
        for date, summary in summaries.items():
            print(f"\n{summary}")
            print("-" * 40)
    elif choice == '2':
        date = input("Enter the date (YYYY-MM-DD): ")
        summary = summarize_day(df, date)
        print(summary)