
import pandas as pd
import io

def analyze_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into runs based on header
    parts = content.split('id,category,prompt,baseline_refused,ablated_refused,baseline_response,ablated_response')
    # The first part might be empty if the file starts with header
    runs = [p for p in parts if p.strip()]
    
    print(f"Found {len(runs)} runs in the file.")
    
    for i, run_content in enumerate(runs):
        print(f"\n--- Run {i+1} ---")
        # Re-add header
        csv_content = 'id,category,prompt,baseline_refused,ablated_refused,baseline_response,ablated_response' + run_content
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            print(f"Total rows: {len(df)}")
            print(f"Baseline Refusal Rate: {df['baseline_refused'].mean():.2%}")
            print(f"Ablated Refusal Rate: {df['ablated_refused'].mean():.2%}")
            
            # Check for "Joy" responses
            joy_responses = df[df['ablated_response'].str.contains("Here's a fun", na=False) | df['ablated_response'].str.contains("The Joy of", na=False)]
            print(f"Rows with 'Joy/Fun' template: {len(joy_responses)}")
            
            # Check how many Joy responses were marked as NOT refused
            joy_not_refused = joy_responses[joy_responses['ablated_refused'] == False]
            print(f"Joy responses marked as NOT refused (False Negatives): {len(joy_not_refused)}")
            
        except Exception as e:
            print(f"Error parsing run {i+1}: {e}")

analyze_csv('batch_test_results_qwen.csv')
