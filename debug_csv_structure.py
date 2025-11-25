import csv
import sys

filename = "jailbreakbench_100.csv"
expected_columns = 4

print(f"--- Scanning {filename} for structure errors ---")

try:
    with open(filename, 'r', encoding='utf-8') as f:
        # Read raw lines first to check for unquoted commas manually if csv module fails
        lines = f.readlines()
        
    print(f"Total lines: {len(lines)}")
    
    # Use csv sniffer to guess dialect, or default to standard
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
    except csv.Error:
        print("Could not sniff dialect, assuming standard CSV.")
        dialect = None

    malformed_count = 0
    
    # We will parse line by line manually to catch the specific lines that fail
    for i, line in enumerate(lines):
        line_num = i + 1
        # Naive split check
        naive_parts = line.strip().split(',')
        
        # If naive split > 4, it MIGHT be an error, unless it's properly quoted.
        if len(naive_parts) > expected_columns:
            # Let's check if it parses correctly as a CSV row
            try:
                reader = csv.reader([line])
                parsed_row = next(reader)
                
                if len(parsed_row) != expected_columns:
                    print(f"[Line {line_num}] ERROR: Found {len(parsed_row)} columns (Expected {expected_columns}).")
                    print(f"    Raw content: {line.strip()[:80]}...")
                    malformed_count += 1
                # If parsed_row has 4 columns but naive has >4, it means quoting is working!
                
            except csv.Error as e:
                print(f"[Line {line_num}] CSV PARSE ERROR: {e}")
                malformed_count += 1

    if malformed_count == 0:
        print("\nSUCCESS: No structural errors found. Pandas should be able to read this.")
    else:
        print(f"\nFAILURE: Found {malformed_count} malformed lines.")

except Exception as e:
    print(f"Critical Error: {e}")
