import config
from MinhashLSH import Minhash
import pickle
from tqdm import tqdm
import time
import random
import json

def train_all(k = '3'):
    type = ['paragraph', 'section', 'phrase']
    for t in type:
        model = Minhash(t, k=k)
        model.train(config.filepath)
        import gc
        gc.collect()

    model = Minhash('trigram',k='3')
    model.train(config.filepath)
    exit()

def testing(all=True,type='',k='3'):
    if all:
        types = ['trigram','paragraph', 'section', 'phrase']
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

        path_test = 'testing_file/_'+t
        with open(path_test, 'rb') as handle:
            queries = pickle.load(handle)

        NUM_TEST = 10

        for i in tqdm(range(1,NUM_TEST + 1)):
            random_index = random.randint(1, len(queries) - 1)
            query = queries[random_index]
            res = model.predict(query,Trigram=T)
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

    # ===== TRAIN ALL ======================================
    config.DEBUG = True
    #  k = { '1', '3'}
    train_all(k='3')

    # ===== PRINT TEST FILE .pickle ========================
    # type = 'section'
    # with open('test_' + type, 'rb') as handle:
    #     l = pickle.load(handle)
    # [ print(i) for i in l]
    # print("TOTAL: {}".format(len(l)))
    # exit()

    # ===== TESTING ========================================
    # type, k = 'phrase', '1'
    # testing(all=False,type=type,k=k)


    # ===== SINGLE TEST =====================================
    # t = "trigram"
    # query = "This Decision will be applicable from this date of publication of the Commission Recommendation"
    # query ="""in addition, the commission will consult member states, the stakeholders and the authority
    #        to discuss the possibility to reduce the current maximum limits in all
    #        meat products and to further simplify the rules for the traditionally manufactured products"""
    # T = False
    # if t == 'trigram':
    #     T = True
    # model = LSH(t,k='1')
    # model.load_lsh()
    # res = model.predict(query, Trigram=T)
    # print(json.dumps(res, ensure_ascii=False, indent=4))


