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
from tqdm import tqdm
import random
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

    def __train_trigram(self,data,file_example=False,part=""):
        start_time = time.time()
        forest = MinHashLSHForest(num_perm=config.permutations)
        num_trigram = 0
        example = []
        for item in tqdm(data):
            num_trigram += 1
            tokens = item[0]['data']
            tag = item[0]['tag']
            if random.randint(0, 1000) == 5:
                item_choice = tag.split("]")[1]
                if len("".join(item_choice.split("]"))) > 8:
                    example += [ item_choice ]

            m = MinHash(num_perm=config.permutations)
            for s in tokens:
                m.update(s.encode('utf8'))
            forest.add(tag,m)
            next(data)

        forest.index()
        print('It took %.2f seconds to build forest.' % (time.time() - start_time))

        time.sleep(1)

        if file_example:
            print("=== Saving on file: test_{} ====".format(part))
            with open('test_'+part, 'wb') as f:
                pickle.dump(example, f)

        return forest


    def __train(self,data,type,file_example=False,part=""):
        start_time = time.time()
        forest = MinHashLSHForest(num_perm=config.permutations)
        example = []
        for item in tqdm(data, desc="MinHash Docs.."):
            tag = item['tag']
            tokens = item['data']
            import random
            if random.randint(0, 100) < 10:
                example += [tag.split("]",1)[1]]
            m = MinHash(num_perm=config.permutations)
            for s in tokens:
                m.update(s.encode('utf8'))
            forest.add(tag,m)

        forest.index()
        print('It took %.2f seconds to build forest.' % (time.time() - start_time))

        time.sleep(1)
        if file_example:
            if file_example:
                print("=== Saving on file: test_{} ====".format(part))
                with open('test_' + part, 'wb') as f:
                    pickle.dump(example, f)

        return forest


    def train(self,dataset_path,part):
        print("====== TRAINING {}...".format(part))
        from preprocess.process_data import Processer
        if not part == "trigram":
            processer = Processer(
                filepath=dataset_path,
                part=part
            )
            data = processer.run()
            lsh = self.__train(data,part,file_example=config.FILE_TEST,part=part)
        else:
            processer = iter(Processer(
                filepath=dataset_path,
                part=part
            ))
            lsh = self.__train_trigram(processer,file_example=config.FILE_TEST,part=part)
        file = config.path_models +"_" + part
        self.__save_lsh(lsh,file)
        print("Model SAVED ~ {}".format(file))
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
            tokens = [ query_norm ]

        start_time = time.time()
        m = MinHash(num_perm=self.permutation)
        for s in tokens:
            m.update(s.encode('utf8'))
        idx_array = np.array(self.model.query(m, N))

        timing_search = "%.2f ms" % ((time.time() - start_time) * 1000)

        if len(idx_array) == 0:
            res_json = []
        else:
            res_json = [
                metrics.metric(query_norm, doc_retrival, self.normalizer,Trigram=Trigram)
                for doc_retrival in idx_array
            ]
            res_json = [
                res
                for res in sorted(res_json, key=lambda i: i['lev'], reverse=True)
                if float(res['lev']) >= threshold
            ]

        timing = "%.2f ms" % ((time.time() - start_time) * 1000)
        # print('It took {} ms to query forest.'.format(timing))

        return {'query': query, 'data': res_json, 'time': timing, 'max':N, 'time_search':timing_search, 'threshold':threshold}



if __name__ == '__main__':
    model = LSH()

    # ===== PRINT TEST FILE .pickle ========================
    # type = 'section'
    # with open('test_' + type, 'rb') as handle:
    #     l = pickle.load(handle)
    # [ print(i) for i in l]
    # print("TOTAL: {}".format(len(l)))
    # exit()

    # ===== TRAIN ==========================================
    # config.DEBUG = True
    # type = 'phrase'
    # model.train(config.filepath, type)
    # import gc
    # gc.collect()
    # exit()

    # ===== SINGLE TEST =====================================
    # query = "This Decision will be applicable from this date of publication of the Commission Recommendation"
    # T = False
    # t = "trigram"
    # if t == 'trigram':
    #     T = True
    # model.load_lsh("./model/model_" + t)
    # res = model.predict(query, Trigram=T)
    # print(json.dumps(res, ensure_ascii=False, indent=4))
    # exit()


    # ===== TESTING ========================================
    for t in ['trigram', 'paragraph', 'section', 'phrase']:
        empty = 0
        T = False
        if t == 'trigram':
            T = True
        type = t
        print("== {} == ".format(t))
        model.load_lsh("./model/model_" + type)
        print("model load!")

        with open('test_' + type, 'rb') as handle:
            queries = pickle.load(handle)
        NUM_TEST = 100
        for i in tqdm(range(1,NUM_TEST + 1)):
            random_index = random.randint(1, len(queries) - 1)
            query = queries[random_index]
            res = model.predict(query,Trigram=T)
            if len(res['data']) == 0:
                empty += 1
            # print(json.dumps(res, ensure_ascii=False, indent=4))
        time.sleep(0.25)
        print("Empty Result: ", empty)

