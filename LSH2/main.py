# http://ethen8181.github.io/machine-learning/recsys/content_based/lsh_text.html
import pandas as pd
import numpy as np

df = pd.read_csv('sample-data.csv')
print('dimension: ', df.shape)
print(df.head())


from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    min_df=0,
    stop_words='english')
X_tfidf = tfidf.fit_transform(df['description'])

def get_similarity_items(X_tfidf, item_id, topn=5):
    """
    Get the top similar items for a given item id.
    The similarity measure here is based on cosine distance.
    """
    query = X_tfidf[item_id]
    print(query)
    scores = X_tfidf.dot(query.T).toarray().ravel()
    best = np.argpartition(scores, -topn)[-topn:]
    return sorted(zip(best, scores[best]), key=lambda x: -x[1])


similar_items = get_similarity_items(X_tfidf, item_id=1)

# an item is always most similar to itself, in real-world
# scenario we might want to filter itself out from the output
for similar_item, similarity in similar_items:
    item_description = df.loc[similar_item, 'description']
    print('similar item id: ', similar_item)
    print('cosine similarity: ', similarity)
    print('item description: ', item_description)
    print()
