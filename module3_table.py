import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate  

df = pd.read_csv("countries of the world.csv")

df["Country"] = df["Country"].astype(str).str.strip().str.replace('"', '').str.lower()
df.replace(",", ".", regex=True, inplace=True)
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.fillna(df.mean(numeric_only=True), inplace=True)
df_clean = df.reset_index(drop=True)


features = df_clean.drop(columns=["Country", "Region"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
cos_sim_matrix = cosine_similarity(scaled_features)
country_to_index = pd.Series(df_clean.index, index=df_clean["Country"])

def print_similar_countries(country_name, top_n=10):
    query = country_name.strip().lower()
    if query not in country_to_index:
        print(f" Country '{country_name}' not found in dataset.")
        return

    idx = country_to_index[query]
    sim_scores = list(enumerate(cos_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sim_scores[1:top_n + 1]]

    result = df_clean.iloc[top_indices][["Country", "Region"]].copy()
    result["Similarity Score"] = [sim_scores[i][1] for i in range(1, top_n + 1)]
    result.insert(0, "Rank", range(1, top_n + 1))

    print(f"\nTop 10 Countries Similar to: {country_name.title()}")
    print(tabulate(result, headers="keys", tablefmt="fancy_grid", showindex=False))

for country in ["kazakhstan", "canada", "morocco"]:
    print_similar_countries(country)
