import config
from MinhashLSH import Minhash
import pickle
from tqdm import tqdm
import time
import random
import json


def testing(all=True,type='',k='3'):
    if all:
        types = ['paragraph', 'section', 'phrase']
    else:
        types = [type]

    for t in types:
        model = Minhash(t, k=k)
        model.load()
        print("model load!")
        empty = 0
        T = False

        print("== {} == ".format(t))
        if t == 'trigram':
            T = True
            # per prendere le frasi come spunto per i trigrammi
            t = 'phrase'

        path_test = '/home/anto/Scrivania/Tesi/testing/testing_file/'+t
        with open(path_test, 'rb') as handle:
            queries = pickle.load(handle)

        NUM_TEST = 10
        queries = queries[:NUM_TEST]

        for q in queries:
            res = model.predict(q,Trigram=T)
            if len(res['data']) == 0:
                empty += 1
            print(json.dumps(res, ensure_ascii=False, indent=4))
        time.sleep(0.25)
        print("Empty Result: ", empty)
        print("=====================",t,k, "===========================")
        import gc
        gc.collect()



    exit()

if __name__ == '__main__':
    # ===== TESTING =======================================
    # type, k = 'phrase', '3'
    # testing()
    # exit()
#     TRIGRAMMM
    query = "asdaohuasf uwofowfejr oerguj eouhlegh le  flrefkd knfkdf kdnwfle feoifleirfelrferieel e lfjsnd lorem lorem council of 22 november 2012 on quality"
    model = Minhash('trigram',k='3')
    model.load()
    res = model.predict(query)
    import json
    print(json.dumps(res,indent=4))