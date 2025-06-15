import pandas as pd
import pickle
import pymysql
import numpy as np
from category_encoders import HashingEncoder

# Load data
conn = pymysql.connect(host='localhost', user='root', password='root@2024', database='swiggy')
mycursor = conn.cursor()
mycursor.execute('SELECT * FROM swiggy')
result = mycursor.fetchall()
df = pd.DataFrame(result, columns=['Id', 'Name', 'City', 'Rating', 'Rating_count', 'Cost', 'Cuisine', 'Lic_No', 'Link', 'Address', 'Menu'])

df = df.drop_duplicates().reset_index(drop=True)


# Clean 'Cost' column
df['Cost'] = df['Cost'].replace(r'[^\d.]', '', regex=True)
df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')

# Ensure numeric types
df['Rating'] = df['Rating'].replace(['--', '', 'NaN'], np.nan)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Rating_count'] = pd.to_numeric(df['Rating_count'], errors='coerce')

# Fill NaNs safely using direct assignment
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['Rating_count'] = df['Rating_count'].fillna(0)
df['Cost'] = df['Cost'].fillna(df['Cost'].median())

# Fill missing strings
df['City'] = df['City'].fillna('Unknown')
df['Cuisine'] = df['Cuisine'].fillna('Others')



# Drop rows with too many NaNs
row_thresh = int(df.shape[1] * 0.6)
df = df.dropna(thresh=row_thresh)

df.to_csv("cleaned_data.csv", index=False)


# Define categorical and numeric columns
categorical_cols = ['City', 'Cuisine']
numeric_cols = ['Rating', 'Rating_count', 'Cost']

# Apply HashingEncoder on categorical columns
encoder = HashingEncoder(cols=categorical_cols, n_components=16)
df_encoded_cat = encoder.fit_transform(df[categorical_cols])

# Combine with numeric columns
df_encoded = pd.concat([df_encoded_cat, df[numeric_cols].reset_index(drop=True)], axis=1)

# Save encoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Save to CSV
df_encoded.to_csv('onehotencoder_data.csv', index=False)

print("Converted the Pickle file into CSV file")
