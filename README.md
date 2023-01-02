## Dependencies
sklearn
pandas
matplotlib
## Components
Cosine Similarity
cosine_similarity from sklearn.metrics.pairwise is used to compare the similarity between the poems.

## Count Vectorizer
CountVectorizer from sklearn.feature_extraction.text is used to convert the poems into numerical vectors for comparison using cosine similarity.

## DataFrame
pandas.DataFrame is used to store the poems and their similarities in a tabular format.

## Matplotlib
matplotlib.pyplot and matplotlib.colors are used to generate a plot to visualize the similarities between the poems.

## Usage
1. Define the poems to compare in the variables poem1, poem2, poem3, poem4, poem5, and poem6.
2. Run the code. The resulting plot will show the similarity between the poems. It will also output a table for the same and save a .csv file with the data.
