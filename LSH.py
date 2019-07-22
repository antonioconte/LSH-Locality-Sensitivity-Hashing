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
    def __init__(self,perm = config.permutations, num_res = config.num_recommendations):
        nlp = spacy.load('en_core_web_sm')
        self.normalizer = TextPipeline(nlp)
        self.permutation = perm
        self.num_results = num_res
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


    def train(self,dataset_path,part,model_path = config.path_models):
        from preprocess.process_data import Processer
        processer = Processer(
            filepath=dataset_path,
            part=part
        )
        # Generazione del formato atteso da LSH.train
        data = processer.run()
        # print(json.dumps(data, indent=4, sort_keys=True))
        lsh = self.__train(data)
        file = model_path +"_" + part
        self.__save_lsh(lsh,file)
        print("Model SAVED ~ {}".format(file))

    def predict(self,query,metric,N=0):
        if self.model == None:
            raise Exception("Model is not loaded!")
        if N == 0:
            N = self.num_results
        query = cleanhtml(query)
        start_time = time.time()
        # senza divisione in ngrammi == False
        tokens = self.normalizer.convert(query, True)
        m = MinHash(num_perm=self.permutation)
        for s in tokens:
            m.update(s.encode('utf8'))

        idx_array = np.array(self.model.query(m, self.num_results))

        timing = "%.2f ms" % ((time.time() - start_time) * 1000)
        print('It took {} ms to query forest.'.format(timing))

        if len(idx_array) == 0:
            res_json = []
        else:
            result = [metrics.metric(query, doc_retrival, self.normalizer, m=metric) for doc_retrival in idx_array]
            if metric == "lev":
                res_json = sorted(result, key=lambda i: i[metric])
            else:
                res_json = sorted(result, key=lambda i: i[metric], reverse=True)

        return {'query': query, 'data': res_json, 'time': timing}


if __name__ == '__main__':
    lsh = LSH()
    config.DEBUG = False
    # lsh.load_lsh("./model/model_"+ "phrase")
    # query = """<p>The opportunities for establishing economic growth through innovation and a sustainable competitive energy policy have been recognised</p>"""
    # res = lsh.predict(query,"lev")
    #
    # print(json.dumps(res,ensure_ascii=False,indent=4))
    # exit(1)

    model_type_train = ["phrase", "paragraph", "section"]

    for m in model_type_train:
        lsh.train(config.filepath, m)
    exit(1)
