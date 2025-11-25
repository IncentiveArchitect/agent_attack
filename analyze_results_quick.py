import pandas as pd

def analyze(filename, model_name):
    try:
        df = pd.read_csv(filename)
        total = len(df)
        baseline_refusals = df['baseline_refused'].sum()
        ablated_refusals = df['ablated_refused'].sum()
        
        print(f"--- {model_name} ---")
        print(f"Total: {total}")
        print(f"Baseline Refusal Rate: {baseline_refusals/total*100:.1f}% ({baseline_refusals}/{total})")
        print(f"Ablated Refusal Rate: {ablated_refusals/total*100:.1f}% ({ablated_refusals}/{total})")
        print("-" * 20)
    except Exception as e:
        print(f"Error analyzing {filename}: {e}")

analyze("batch_test_results_qwen.csv", "Qwen1.5-1.8B-Chat")
analyze("batch_test_results.csv", "Llama-3-8B-Instruct")
