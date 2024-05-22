import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../data/imdb-movies-dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Plot distribution of sentiments
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()

# Display some example reviews
for idx, row in df.sample(5).iterrows():
    print(f"Review: {row['review']}")
    print(f"Sentiment: {row['sentiment']}\n")
