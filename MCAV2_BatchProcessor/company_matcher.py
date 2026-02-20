import pandas as pd
import os
from fuzzywuzzy import fuzz

# Read the CSV file with encoding fix for special characters
try:
    df = pd.read_csv(r"D:\Dev\Github\MCAV2\data\training_dataset.csv", encoding='ISO-8859-1')
except UnicodeDecodeError:
    df = pd.read_csv(r"D:\Dev\Github\MCAV2\data\training_dataset.csv", encoding='cp1252')

# Folder path
json_folder = r"D:\Dev\Github\MCAV2\data\JsonExport"

# Get IDs from the application_id column
companies = df['application_id'].dropna().tolist()

# Get actual filenames from your folder
if os.path.exists(json_folder):
    files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    # NEW: Print all JSON files found in the folder
    print(f"--- JSON FILES FOUND IN FOLDER ({len(files)}) ---")
    for i, file_name in enumerate(sorted(files), 1):
        print(f"{i}. {file_name}")
    print("-" * 50)
else:
    print(f"Error: Folder not found at {json_folder}")
    files = []

def normalize_name(name):
    name = str(name).lower().strip()
    name = name.replace('.json', '').replace(' limited', ' ltd').replace(' ltd.', ' ltd')
    return name

missing_companies = []

if not files:
    print("\nNo files found to compare against!")
    exit()

print("\nStarting comparison...")
for company in companies:
    norm_company = normalize_name(company)
    
    best_score = 0
    best_match = ""
    for file in files:
        norm_file = normalize_name(file)
        score = max(
            fuzz.ratio(norm_company, norm_file),
            fuzz.token_set_ratio(norm_company, norm_file)
        )
        if score > best_score:
            best_score = score
            best_match = file
    
    if best_score < 90:
        missing_companies.append(company)
        print(f"MISSING: {company} (Best guess: {best_match} - {best_score}%)")

# Make sure the lines below are NOT indented (they should be flush with the left margin)
print(f"\n" + "="*30)
print("SUMMARY:")
print("="*30)
print(f"Total IDs in CSV: {len(companies)}")
print(f"Total JSON files: {len(files)}")
print(f"Missing (no match): {len(missing_companies)}")

# Save missing companies to file
with open('missing_companies.txt', 'w') as f:
    for company in missing_companies:
        f.write(str(company) + '\n')

print(f"\nMissing IDs saved to 'missing_companies.txt'")