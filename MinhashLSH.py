from datasketch import MinHash, MinHashLSHForest
from preprocess.text_pipeline import TextPipeline
from preprocess.utils import cleanhtml
from utils import metrics
from tqdm import tqdm
import config
import json
import pickle
import spacy
import time
import random
import numpy as np

class Minhash():
    def __init__(self,type, k = '3'):
        nlp = spacy.load('en_core_web_'+config.size_nlp)
        self.k = k
        self.entryname = 'Minhash_K_'+self.k
        config.kGRAM = self.k
        self.normalizer = TextPipeline(nlp)
        self.permutation = config.permutations
        self.type = type
        self.model = None
        self.path_model = ""
        if self.type in 'trigram':
            self.path_model = config.path_models + "_" + self.type
        else:
            self.path_model = config.path_models + "_" + self.type + "_" + self.k

        self.path_test = 'testing_file/' + "_" + self.type
        self.pathDataProc = config.pathDataProc.format(self.type,self.k)


    def __save(self, obj, path="model"):
        with open(self.path_model, 'wb') as f:
            pickle.dump(obj, f)
        print("Saved: {}".format(path))

    def load(self):
        with (open(self.path_model, "rb")) as f:
            self.model = pickle.load(f)

    def __train_trigram(self,data):
        start_time = time.time()
        forest = MinHashLSHForest(num_perm=config.permutations)
        num_trigram = 0
        for item in data:
            try:
                if len(item[0]['data'][0].split()) < 3:
                    continue
                num_trigram += 1
                tokens = item[0]['data']
                tag = item[0]['tag']
                m = MinHash(num_perm=config.permutations)
                for s in tokens:
                    m.update(s.encode('utf8'))
                forest.add(tag,m)
            except:
                # print('err:',string,tokens)
                continue

        print("Num Trigrams: {}".format(num_trigram))

        forest.index()
        print('It took %.2f seconds to build forest.' % (time.time() - start_time))

        time.sleep(1)

        return forest


    def __train(self,data):
        start_time = time.time()
        forest = MinHashLSHForest(num_perm=config.permutations)
        for item in tqdm(data, desc="MinHash Docs.."):
            tag = item['tag']
            tokens = item['data']

            m = MinHash(num_perm=config.permutations)
            for s in tokens:
                m.update(s.encode('utf8'))
            forest.add(tag,m)

        forest.index()
        print('It took %.2f seconds to build forest.' % (time.time() - start_time))
        return forest


    def train(self,dataset_path):
        part = self.type
        print("====== TRAINING {} [ K = {} ] ...".format(part,config.kGRAM))
        from preprocess.process_data import Processer
        if not part == "trigram":
           with open(self.pathDataProc, 'rb') as handle:
                data = pickle.load(handle)
                m_minhash = self.__train(data)
        else:
            processer = iter(Processer(
                filepath=dataset_path,
                part=part
            ))
            m_minhash = self.__train_trigram(processer)


        self.__save(m_minhash,self.path_model)
        print("Model SAVED ~ {}".format(self.path_model))
        print("================================")


    def predict(self,
                query,
                threshold=config.default_threshold,
                N=config.num_recommendations,
                Trigram = False):
        if self.model == None:
            raise Exception("Model is not loaded!")

        query = cleanhtml(query)

        if not Trigram:
            query_norm = self.normalizer.convert(query,False)
            # True per la fase di predict
            tokens = self.normalizer.convert(query, divNGram=True)
        else:
            query, query_norm = self.normalizer.norm_text_trigram(query)
            if query_norm == None:
                return {'query': query, 'data': [], 'time': '0 ms', 'max': N, 'time_search': '0 ms',
                        'threshold': threshold}
            else:
                tokens = [ query_norm ]

        start_time = time.time()
        m = MinHash(num_perm=self.permutation)
        for s in tokens:
            m.update(s.encode('utf8'))

        # m è la query sotto forma di bucket ed N è il numero max di elementi richiesti
        idx_array = np.array(self.model.query(m, N))

        timing_search = "%.2f ms" % ((time.time() - start_time) * 1000)

        if len(idx_array) == 0:
            res_json = []
        else:
            res_json = []
            for doc_retrival in idx_array:
                item = metrics.metric(query_norm, doc_retrival, self.normalizer,Trigram=Trigram)
                if float(item['lev']) >= threshold:
                    res_json += [item]
            # ====== RE-RANKING =========================================================
            res_json = sorted(res_json, key=lambda i: i['lev'], reverse=True)

        timing = "%.2f ms" % ((time.time() - start_time) * 1000)
        return {'query': query, 'data': res_json, 'time': timing, 'max':N, 'time_search':timing_search, 'threshold':threshold, 'algoritm':self.entryname}