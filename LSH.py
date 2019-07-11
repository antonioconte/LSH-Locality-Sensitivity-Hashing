import numpy as np
from preprocess import utils
from datasketch import MinHash, MinHashLSHForest
import pickle
import textdistance
import time

def save_lsh(obj, path="model"):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print("Saved: {}".format(path))

def load_lsh(path="model"):
    with (open(path, "rb")) as f:
        lsh = pickle.load(f)
    return lsh

def train(data, perms):
    start_time = time.time()
    minhash = []
    for item in data:
        # tag = item['tag']
        tokens = item['data']
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)

    forest = MinHashLSHForest(num_perm=perms)
    for i, m in enumerate(minhash):
        forest.add(data[i]['tag'], m)

    forest.index()

    print('It took %.2f seconds to build forest.' % (time.time() - start_time))

    return forest

def metric(query, doc, m=""):
    (tag,text) = doc.split("]")
    if m == "jac":
        jac = textdistance.Jaccard()
        value = "%.2f" % jac(query,text)
    elif m == "lev":
        lev = textdistance.Levenshtein()
        value = str(lev.distance(query,text))
    else:
        value = 'NaN'

    return {'docname': tag[1:].split("#")[0],'text': text, m: value}
    # return {'text': tag, m: value}

def predict(text, perms, num_results, forest):
    # METRICS = "jac"
    METRICS = "lev"

    print("METRICA",METRICS)

    start_time = time.time()
    tokens = utils.preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))

    idx_array = np.array(forest.query(m, num_results))

    timing = "%.2f ms" % ((time.time() - start_time) * 1000)
    print('It took {} ms to query forest.'.format(timing))

    if len(idx_array) == 0:
        res_json = []
    else:
        result = [ metric(text,doc_retrival,m=METRICS) for doc_retrival in idx_array]
        if METRICS == "lev":
            res_json = sorted(result, key = lambda i: int(i[METRICS]))
        else:
            res_json = sorted(result, key = lambda i: i[METRICS], reverse=True)

    return {'query': text,'data': res_json, 'time': timing }



if __name__ == '__main__':
    permutations = 128
    num_recommendations = 5
