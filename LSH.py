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

    def predict(self,query,threshold=0.75,N=0):
        print("> QUERY:",query)
        if self.model == None:
            raise Exception("Model is not loaded!")
        if N == 0:
            N = self.num_results

        query = cleanhtml(query)
        start_time = time.time()
        # True per la fase di predict
        tokens = self.normalizer.convert(query, True)
        print("> TOKENS:",tokens)

        m = MinHash(num_perm=self.permutation)
        for s in tokens:
            m.update(s.encode('utf8'))

        idx_array = np.array(self.model.query(m, N))

        if len(idx_array) == 0:
            res_json = []
        else:
            result = [metrics.metric(query, doc_retrival, self.normalizer, m='lev_sim') for doc_retrival in idx_array]
            res_json = [ res for res in sorted(result, key=lambda i: i['lev'], reverse=True) if float(res['lev']) >= threshold]
            # print(list(filter(lambda x: x['lev_sim'] > 0.8, res_json)))

        timing = "%.2f ms" % ((time.time() - start_time) * 1000)
        print('It took {} ms to query forest.'.format(timing))

        return {'query': query, 'data': res_json, 'time': timing, 'max':N, 'threshold':threshold}


if __name__ == '__main__':
    lsh = LSH()
    config.DEBUG = False
    lsh.load_lsh("./model/model_"+ "phrase")
    query = """<p>This Decision will be applicable from this date of publication of the Commission Recommendation</p>"""
    res = lsh.predict(query,threshold=0.51)

    print(json.dumps(res,ensure_ascii=False,indent=4))
    exit(1)

    model_type_train = ["phrase", "paragraph", "section"]

    for m in model_type_train:
        lsh.train(config.filepath, m)
    exit(1)
