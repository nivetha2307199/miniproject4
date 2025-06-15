# miniproject4
Swiggy Restaurant Recommender
Step1:Data Cleaning:-1)Cleaned all the empty data
                    2)converted string columns into numerical columns
                    3)removed the categorial(unknown data or others) in some String columns(City and cuisines)
Step2:-Pickle file conversion:-1)Read the cleaned data
                    2)Apply One-Hot Encoding to categorical features: Restaurant Name, City, and Cuisine
                    3)Ensure all features are numeric after encoding
                    4)Save the encoder to encoder.pkl
                    5)Save the encoded dataset to onehotencoded_data.csv
                    6)Ensure index alignment with cleaned_data.csv
Step3:-Recommendation Methodology:-1)Fit K-Means on the encoded dataset (encoded_data.csv)
                    2)Predict the cluster of the user input (after encoding it)
                    3)Recommend restaurants from the same cluster
Step4:-Used Streamlit UI:-1)User Input Section:{City (dropdown),Cuisine (dropdown),Cost (slider),Rating (slider)}
                    2) Recommendation Engine
                       1)Encode user input using encoder.pkl
                       2)Predict cluster (KMeans) or compute similarity (Cosine)
                       3)Recommend matching restaurants from cleaned_data.csv
                    3)Output
                        1)Show results in a clean table (name, cuisine, price, rating, city,Address)
