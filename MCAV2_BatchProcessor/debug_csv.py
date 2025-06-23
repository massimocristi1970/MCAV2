import pandas as pd

# Read the CSV file
csv_path = r"C:\Users\Massimo Cristi\OneDrive\Documents\GitHub\MCAv2\MCAV2_BatchProcessor\data\applicationsdata.csv"

try:
    df = pd.read_csv(csv_path)
    print("CSV loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    print("\n" + "="*50)
    print("COLUMN A DATA (first 10):")
    print("="*50)
    for i, value in enumerate(df.iloc[:10, 0]):
        print(f"Row {i+1}: '{value}'")
    
    print("\n" + "="*50)
    print("COLUMN B DATA (first 10):")
    print("="*50)
    for i, value in enumerate(df.iloc[:10, 1]):
        print(f"Row {i+1}: '{value}'")
    
    print("\n" + "="*50)
    print("DATA TYPES:")
    print("="*50)
    print(df.dtypes)
    
    print("\n" + "="*50)
    print("NULL VALUES:")
    print("="*50)
    print(df.isnull().sum())
    
except Exception as e:
    print(f"Error reading CSV: {e}")
    print("Let's check if the file exists...")
    import os
    if os.path.exists(csv_path):
        print("File exists!")
        # Try reading first few lines as text
        with open(csv_path, 'r', encoding='utf-8') as f:
            print("First 5 lines of file:")
            for i, line in enumerate(f):
                if i < 5:
                    print(f"Line {i+1}: {repr(line)}")
                else:
                    break
    else:
        print("File does not exist at that path!")