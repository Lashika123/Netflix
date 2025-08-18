import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('netflix_titles.csv')

# Drop duplicates
df = df.drop_duplicates()

# Drop rows with missing essential fields
df = df.dropna(subset=['title', 'type', 'release_year'])

# Strip whitespace and standardize case in key string columns
cat_cols = ['type', 'rating', 'country', 'director', 'cast', 'listed_in']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).replace(r'^\s*$', np.nan, regex=True)
        df[col] = df[col].apply(lambda x: x.strip().title() if isinstance(x, str) else x)

# Convert 'release_year' to numeric (already likely is)
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

# Convert 'date_added' to datetime; create 'year_added'
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

# Replace blank fields with NaN everywhere
df = df.replace(r'^\s*$', np.nan, regex=True)

# Save cleaned dataset
df.to_csv('netflix_titles_cleaned.csv', index=False)
print("✅ Cleaned file saved as netflix_titles_cleaned.csv")
