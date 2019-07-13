import numpy as np
from preprocess import utils
import time
from datasketch import MinHash, MinHashLSHForest
import pickle

def save_lsh(obj, path="model"):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print("Saved: {}".format(path))


def load_lsh(path="model"):
    with (open(path, "rb")) as f:
        lsh = pickle.load(f)
    return lsh

def get_forest(data, perms):
    start_time = time.time()
    minhash = []
    for text in data['text']:
        tokens = utils.preprocess(text)
        print(text)
        print(tokens)
        print()
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)

    forest = MinHashLSHForest(num_perm=perms)

    for i, m in enumerate(minhash):
        forest.add(i, m)

    forest.index()

    print('\nIt took %.2f seconds to build forest.' % (time.time() - start_time))

    return forest


def predict(text, database, perms, num_results, forest):
    start_time = time.time()

    tokens = utils.preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))

    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None  # if your query is empty, return none

    # result = idx_array
    result = database.iloc[idx_array]['title']

    print('It took %.4f ms to query forest.' % ((time.time() - start_time) * 1000))

    return result

if __name__ == '__main__':
    permutations = 128
    num_recommendations = 5