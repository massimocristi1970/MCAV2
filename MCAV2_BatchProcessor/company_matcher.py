import pandas as pd
from fuzzywuzzy import fuzz

# Read the CSV file
df = pd.read_csv(r"C:\Users\Massimo Cristi\OneDrive\Documents\GitHub\MCAv2\MCAV2_BatchProcessor\data\applicationsdata.csv")

# Get company names and file names (assuming columns A and B)
companies = df.iloc[:, 0].dropna().tolist()  # Column A
files = df.iloc[:, 1].dropna().tolist()      # Column B

def normalize_name(name):
    name = str(name).lower().strip()
    name = name.replace('.json', '').replace(' limited', ' ltd').replace(' ltd.', ' ltd')
    return name

missing_companies = []

for company in companies:
    norm_company = normalize_name(company)
    
    # Check if any file matches well enough
    best_score = 0
    for file in files:
        norm_file = normalize_name(file)
        score = max(
            fuzz.ratio(norm_company, norm_file),
            fuzz.token_set_ratio(norm_company, norm_file)
        )
        best_score = max(best_score, score)
    
    if best_score < 90:  # 90% similarity threshold
        missing_companies.append(company)
        print(f"MISSING: {company} (best match score: {best_score}%)")

print(f"\nSUMMARY:")
print(f"Total companies: {len(companies)}")
print(f"Missing files: {len(missing_companies)}")
print(f"Files found: {len(companies) - len(missing_companies)}")

# Save missing companies to file
with open('missing_companies.txt', 'w') as f:
    for company in missing_companies:
        f.write(company + '\n')

print(f"\nMissing companies saved to 'missing_companies.txt'")