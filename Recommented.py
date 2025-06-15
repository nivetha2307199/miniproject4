import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

# Load data
cleaned_df = pd.read_csv('cleaned_data.csv')
encoded_df = pd.read_csv('onehotencoder_data.csv')


# Load encoder
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Fit KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
encoded_df['cluster'] = kmeans.fit_predict(encoded_df)

# Assign cluster to cleaned data
cleaned_df['cluster'] = encoded_df['cluster']

# Streamlit UI
st.title("🍽️ Restaurant Recommendation System")
cleaned_df['City'] = cleaned_df['City'].str.strip().str.lower()
cleaned_df['Cuisine'] = cleaned_df['Cuisine'].str.strip().str.lower()

# Ensure Rating and Cost are numeric
cleaned_df['Rating'] = pd.to_numeric(cleaned_df['Rating'], errors='coerce')
cleaned_df['Cost'] = pd.to_numeric(cleaned_df['Cost'], errors='coerce')


# User Input
city = st.selectbox("Select your City", cleaned_df['City'].unique())
cuisine = st.selectbox("Choose a Cuisine", cleaned_df['Cuisine'].explode().str.split(',').explode().str.strip().unique())
cost = st.slider("Max Price for One", 50, 1000, 300)
rating = st.slider("Minimum Rating", 1.0, 5.0, 4.0, 0.1)
city = city.strip().lower()
cuisine = cuisine.strip().lower()


# Recommendation button
if st.button("Get Recommendations"):
    try:
        # Categorical Input
        input_cat = pd.DataFrame([{'City': city, 'Cuisine': cuisine}])
        encoded_input = encoder.transform(input_cat)

        # Convert to dense array
        if hasattr(encoded_input, 'toarray'):
            encoded_input = encoded_input.toarray()
        input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out())

        
        input_df['Cost'] = cost
        input_df['Rating'] = rating

        
        model_cols = encoded_df.drop(columns=['cluster']).columns
        input_df = input_df.reindex(columns=model_cols, fill_value=0)
        cluster = kmeans.predict(input_df)[0]
            
        filtered_df = cleaned_df[cleaned_df['cluster'] == cluster]
        
        filtered_df = cleaned_df[
            (cleaned_df['City'] == city) &
            (cleaned_df['Cuisine'].str.contains(cuisine)) &
            (cleaned_df['Cost'] <= cost)&
            (cleaned_df['Rating']<=rating)
        ]
        # Output
        if not filtered_df.empty:
            st.success("✅ Recommended Restaurants:")
            st.dataframe(filtered_df[['Name', 'City', 'Cuisine', 'Rating', 'Cost','Address']])
        else:
            st.warning("No matching restaurants found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
