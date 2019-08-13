import numpy as np
from utils import metrics
import config
import json
from tqdm import tqdm
from datasketch import MinHash, MinHashLSHForest
import pickle
from preprocess.utils import cleanhtml
import spacy
import time
from preprocess.text_pipeline import TextPipeline

class LSH():
    def __init__(self,perm = config.permutations):
        nlp = spacy.load('en_core_web_'+config.size_nlp)
        self.normalizer = TextPipeline(nlp)
        self.permutation = perm
        self.model = None

    def setPerm(self,p):
        self.permutation = p

    def setNumResults(self,n):
        self.num_results = n

    def __save_lsh(self, obj, path="model"):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print("Saved: {}".format(path))

    def load_lsh(self,path):
        with (open(path, "rb")) as f:
            lsh = pickle.load(f)
        self.model = lsh

    def __train(self,data):
        start_time = time.time()
        minhash = []

        for item in tqdm(data, desc="MinHash Docs.."):
            # tag = item['tag']
            tokens = item['data']
            m = MinHash(num_perm=config.permutations)
            for s in tokens:
                m.update(s.encode('utf8'))
            minhash.append(m)

        forest = MinHashLSHForest(num_perm=config.permutations)
        for i, m in enumerate(minhash):
            forest.add(data[i]['tag'], m)
        forest.index()
        print('It took %.2f seconds to build forest.' % (time.time() - start_time))

        return forest


    def train(self,dataset_path,part):
        from preprocess.process_data import Processer
        print("WORD BASED: ",config.wordBased)
        processer = Processer(
            filepath=dataset_path,
            part=part
        )
        # Generazione del formato atteso da LSH.train
        data = processer.run()
        # print(json.dumps(data[0]['data'],indent=4))
        # print(json.dumps(data, indent=4, sort_keys=True))
        lsh = self.__train(data)
        file = config.path_models +"_" + part
        self.__save_lsh(lsh,file)
        print("Model SAVED ~ {}".format(file))

    def predict(self,query,threshold=config.default_threshold,N=config.num_recommendations):
        if self.model == None:
            raise Exception("Model is not loaded!")

        query = cleanhtml(query)
        query_norm = self.normalizer.convert(query,False)
        start_time = time.time()

        # True per la fase di predict
        tokens = self.normalizer.convert(query, divNGram=True,wordBased=config.wordBased)
        m = MinHash(num_perm=self.permutation)
        for s in tokens:
            m.update(s.encode('utf8'))
        print(tokens)
        idx_array = np.array(self.model.query(m, N))

        timing_search = "%.2f ms" % ((time.time() - start_time) * 1000)

        if len(idx_array) == 0:
            res_json = []
        else:
            res_json = [
                metrics.metric(query_norm, doc_retrival, self.normalizer)
                for doc_retrival in idx_array
            ]
            res_json = [
                res
                for res in sorted(res_json, key=lambda i: i['lev'], reverse=True)
                if float(res['lev']) >= threshold
            ]

        timing = "%.2f ms" % ((time.time() - start_time) * 1000)
        print('It took {} ms to query forest.'.format(timing))

        return {'query': query, 'data': res_json, 'time': timing, 'max':N, 'time_search':timing_search, 'threshold':threshold}


def predict():
    lsh.load_lsh("./model/model_" + "trigram")
    query = "journal european union"
    query = "european parliament council"
    res = lsh.predict(query)
    print(json.dumps(res, ensure_ascii=False, indent=4))
    exit(1)
if __name__ == '__main__':
    lsh = LSH()
    predict()


    # model_type_train = ["paragraph"]
    # model_type_train = ["phrase"]
    # model_type_train = ["section"]
    model_type_train = ["trigram"]

    # model_type_train = ["paragraph", "section","phrase","trigram"]
    for m in model_type_train:
        lsh.train(config.filepath, m)
    exit(1)
