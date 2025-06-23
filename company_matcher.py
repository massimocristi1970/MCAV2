import pandas as pd
import os
from fuzzywuzzy import fuzz
import re

def normalize_name(name):
    """Normalize company names for better matching"""
    if pd.isna(name) or name == "":
        return ""
    
    name = str(name).strip().lower()
    
    # Remove file extensions
    name = re.sub(r'\.(json|txt|pdf)$', '', name)
    
    # Standardize company suffixes
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
csv_path = r"C:\Users\Massimo Cristi\OneDrive\Documents\GitHub\MCAv2\MCAV2_BatchProcessor\data\applicationsdata.csv"
json_folder = r"C:\Users\Massimo Cristi\OneDrive\Documents\GitHub\MCAv2\MCAV2_BatchProcessor\data\json_files"

print("Loading company names from CSV...")
df = pd.read_csv(csv_path)
companies = df['company_name'].dropna().tolist()
print(f"Found {len(companies)} companies in CSV")

print(f"\nLooking for JSON files in: {json_folder}")
if os.path.exists(json_folder):
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files")
else:
    print("JSON folder not found! Please check the path.")
    print("Available folders:")
    data_folder = r"C:\Users\Massimo Cristi\OneDrive\Documents\GitHub\MCAv2\MCAV2_BatchProcessor\data"
    if os.path.exists(data_folder):
        for item in os.listdir(data_folder):
            if os.path.isdir(os.path.join(data_folder, item)):
                print(f"  - {item}")
    exit()

print(f"\nFirst 10 JSON files:")
for i, file in enumerate(json_files[:10]):
    print(f"  {i+1}. {file}")

print(f"\nFirst 10 companies:")
for i, company in enumerate(companies[:10]):
    print(f"  {i+1}. {company}")

# Normalize names for comparison
normalized_companies = [normalize_name(comp) for comp in companies]
normalized_files = [normalize_name(file) for file in json_files]

print(f"\nMatching companies with files (threshold: 90%)...")
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
            'company': company,
            'file': best_file,
            'score': best_score
        })
        print(f"✓ MATCH: {company} → {best_file} ({best_score}%)")
    else:
        missing_companies.append(company)
        print(f"✗ MISSING: {company} (best: {best_file} - {best_score}%)")

print(f"\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Total companies: {len(companies)}")
print(f"Files found: {len(found_matches)}")
print(f"Missing files: {len(missing_companies)}")
print(f"Match rate: {len(found_matches)/len(companies)*100:.1f}%")

if missing_companies:
    print(f"\nMISSING COMPANIES:")
    print("-" * 30)
    for company in missing_companies:
        print(f"  - {company}")
    
    # Save missing companies to file
    with open('missing_companies.txt', 'w') as f:
        for company in missing_companies:
            f.write(company + '\n')
    print(f"\nMissing companies saved to 'missing_companies.txt'")

# Save detailed results
results_df = pd.DataFrame(found_matches + [{'company': comp, 'file': 'MISSING', 'score': 0} for comp in missing_companies])
results_df.to_csv('matching_results.csv', index=False)
print(f"Detailed results saved to 'matching_results.csv'")