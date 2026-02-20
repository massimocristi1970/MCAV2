import pandas as pd
import os
from fuzzywuzzy import fuzz

# Read the CSV file
df = pd.read_csv(r"D:\Dev\Github\MCAV2\data\training_dataset.csv")

# UPDATE: Point this to your actual JSON folder
json_folder = r"D:\Dev\Github\MCAV2\data\JsonExport"

# Get IDs from the first column (application_id)
companies = df['application_id'].dropna().tolist()

# Get actual filenames from your folder instead of Column B
if os.path.exists(json_folder):
    files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
else:
    print(f"Error: Folder not found at {json_folder}")
    files = []

def normalize_name(name):
    name = str(name).lower().strip()
    name = name.replace('.json', '').replace(' limited', ' ltd').replace(' ltd.', ' ltd')
    return name

missing_companies = []

if not files:
    print("No files found to compare against!")
    exit()

for company in companies:
    norm_company = normalize_name(company)
    
    best_score = 0
    for file in files:
        norm_file = normalize_name(file)
        score = max(
            fuzz.ratio(norm_company, norm_file),
            fuzz.token_set_ratio(norm_company, norm_file)
        )
        best_score = max(best_score, score)
    
    if best_score < 90:
        missing_companies.append(company)
        print(f"MISSING: {company} (best match score: {best_score}%)")

print(f"\nSUMMARY:")
print(f"Total entries in CSV: {len(companies)}")
print(f"Files found in folder: {len(files)}")
print(f"Missing (no match): {len(missing_companies)}")

# Save missing companies to file
with open('missing_companies.txt', 'w') as f:
    for company in missing_companies:
        f.write(str(company) + '\n')

print(f"\nMissing IDs saved to 'missing_companies.txt'")