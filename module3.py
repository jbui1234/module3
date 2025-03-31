import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("countries of the world.csv")

# Clean and normalize 'Country' column (strip, remove quotes, lowercase)
df["Country"] = df["Country"].astype(str).str.strip().str.replace('"', '').str.lower()

# Replace commas with dots for decimal numbers
df.replace(",", ".", regex=True, inplace=True)

# Convert numeric columns starting from index 2
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values with the column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

df_clean = df.reset_index(drop=True)

features = df_clean.drop(columns=["Country", "Region"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Compute cosine similarity matrix
cos_sim_matrix = cosine_similarity(scaled_features)

country_to_index = pd.Series(df_clean.index, index=df_clean["Country"])

# Function to retrieve top N similar countries
def get_similar_countries(country_name, top_n=10):
    query = country_name.strip().lower()
    if query not in country_to_index:
        return f"Country '{country_name}' not found in dataset."
    
    idx = country_to_index[query]
    sim_scores = list(enumerate(cos_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sim_scores[1:top_n + 1]]  # exclude self

    result = df_clean.iloc[top_indices][["Country", "Region"]].copy()
    result["Similarity Score"] = [sim_scores[i][1] for i in range(1, top_n + 1)]
    result.insert(0, "Rank", range(1, top_n + 1))
    return result

query_countries = ["kazakhstan", "canada", "morocco"]
for country in query_countries:
    print("\n" + "=" * 70)
    print(f"Top 10 Countries Similar to: {country.capitalize()}")
    print("=" * 70)
    result = get_similar_countries(country)
    if isinstance(result, str):
        print(result)
    else:
        print(result.to_string(index=False))
