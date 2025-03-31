import matplotlib.pyplot as plt

similarity_data = {
    "Kazakhstan": [
        ("argentina", 0.865606),
        ("bosnia & herzegovina", 0.777144),
        ("armenia", 0.760497),
        ("uruguay", 0.735558),
        ("chile", 0.725779),
        ("georgia", 0.692647),
        ("belarus", 0.624027),
        ("estonia", 0.607073),
        ("macedonia", 0.578550),
        ("liechtenstein", 0.510599)
    ],
    "Canada": [
        ("australia", 0.939949),
        ("united states", 0.873703),
        ("russia", 0.867593),
        ("brazil", 0.806514),
        ("argentina", 0.637432),
        ("sweden", 0.478925),
        ("switzerland", 0.448202),
        ("norway", 0.447206),
        ("finland", 0.444853),
        ("iceland", 0.433584)
    ],
    "Morocco": [
        ("nepal", 0.779250),
        ("nicaragua", 0.776532),
        ("cambodia", 0.735863),
        ("iraq", 0.725546),
        ("pakistan", 0.658156),
        ("papua new guinea", 0.645351),
        ("senegal", 0.632430),
        ("bhutan", 0.626777),
        ("togo", 0.575215),
        ("gambia. the", 0.572359)
    ]
}

# Bar plots for each query country
for country, data in similarity_data.items():
    labels, scores = zip(*data)
    plt.figure(figsize=(10, 5))
    plt.barh(labels[::-1], scores[::-1])  
    plt.title(f"Top 10 Countries Similar to {country}")
    plt.xlabel("Cosine Similarity Score")
    plt.tight_layout()
    plt.show()

# Scatter plot with point labels
for country, data in similarity_data.items():
    labels, scores = zip(*data)
    x_vals = range(1, 11)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(x_vals, scores, color='teal')
    
    for x, y, label in zip(x_vals, scores, labels):
        plt.text(x + 0.1, y, label, fontsize=8)
    
    plt.title(f"Cosine Similarity: Top 10 Matches for {country}")
    plt.xlabel("Rank")
    plt.ylabel("Similarity Score")
    plt.xticks(range(1, 11))
    plt.ylim(0.4, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

