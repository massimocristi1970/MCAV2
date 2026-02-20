import pandas as pd
import os
from fuzzywuzzy import fuzz
import re

def normalize_name(name):
    """Normalize names for better matching"""
    if pd.isna(name) or name == "":
        return ""
    
    name = str(name).strip().lower()
    
    # Remove file extensions
    name = re.sub(r'\.(json|txt|pdf)$', '', name)
    
    # Standardize common business terms
    replacements = {
        ' limited': ' ltd',
        ' incorporated': ' inc',
        ' corporation': ' corp',
        ' company': ' co',
        ' ltd.': ' ltd',
        ' inc.': ' inc',
        ' corp.': ' corp'
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

# File paths
csv_path = r"D:\Dev\Github\MCAV2\data\training_dataset.csv"
json_folder = r"D:\Dev\Github\MCAV2\data\JsonExport"

print("Loading data from CSV...")
df = pd.read_csv(csv_path)

# UPDATED: Using 'application_id' as the identifier to match against filenames
# If you actually wanted to match by a different column, change 'application_id' below
target_column = 'application_id' 

if target_column not in df.columns:
    print(f"Error: Column '{target_column}' not found in CSV.")
    print(f"Available columns: {list(df.columns)}")
    exit()

companies = df[target_column].dropna().astype(str).tolist()
print(f"Found {len(companies)} entries in CSV under '{target_column}'")

print(f"\nLooking for JSON files in: {json_folder}")
if os.path.exists(json_folder):
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files")
else:
    print("JSON folder not found! Please check the path.")
    exit()

# Normalize names for comparison
normalized_companies = [normalize_name(comp) for comp in companies]
normalized_files = [normalize_name(file) for file in json_files]

print(f"\nMatching '{target_column}' with files (threshold: 90%)...")
missing_companies = []
found_matches = []

for i, company in enumerate(companies):
    norm_company = normalized_companies[i]
    
    best_match = None
    best_score = 0
    best_file = None
    
    for j, file in enumerate(json_files):
        norm_file = normalized_files[j]
        
        # Calculate similarity scores
        score1 = fuzz.ratio(norm_company, norm_file)
        score2 = fuzz.partial_ratio(norm_company, norm_file)
        score3 = fuzz.token_set_ratio(norm_company, norm_file)
        
        score = max(score1, score2, score3)
        
        if score > best_score:
            best_score = score
            best_match = norm_file
            best_file = file
    
    if best_score >= 90:
        found_matches.append({
            'identifier': company,
            'file': best_file,
            'score': best_score
        })
        print(f"✓ MATCH: {company} → {best_file} ({best_score}%)")
    else:
        missing_companies.append(company)
        # Showing the best attempt even if it failed the threshold
        print(f"✗ MISSING: {company} (Best guess: {best_file} - {best_score}%)")

print(f"\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Total entries: {len(companies)}")
print(f"Matches found: {len(found_matches)}")
print(f"Missing files: {len(missing_companies)}")

if len(companies) > 0:
    print(f"Match rate: {len(found_matches)/len(companies)*100:.1f}%")

# Save results
results_df = pd.DataFrame(found_matches + [{'identifier': comp, 'file': 'MISSING', 'score': 0} for comp in missing_companies])
results_df.to_csv('matching_results.csv', index=False)
print(f"\nDetailed results saved to 'matching_results.csv'")