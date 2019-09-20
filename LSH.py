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
        with open(path+ "_" + config.kGRAM, 'wb') as f:
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
        for item in data:
            try:
                string = item[0]['data']
                tokens = string[0].split()
                if len(tokens) < 3:
                    # print('> min of 3',tokens)
                    continue
                num_trigram += 1
                tokens = [tokens[0] + " " + tokens[1], tokens[1] + " " + tokens[2]]

                # tokens = [tokens[0:2],tokens[1:]]
                tag = item[0]['tag']
                if random.randint(0, 1000) == 5:
                    item_choice = tag.split("]")[1]
                    if len("".join(item_choice.split("]"))) > 8:
                        example += [ item_choice ]

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

        if file_example:
            print("=== Saving on file: test_{} ====".format(part))
            with open('test_'+part, 'wb') as f:
                pickle.dump(example, f)

        return forest


    def __train(self,data,file_example=False,part=""):
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
            lsh = self.__train(data,file_example=config.FILE_TEST,part=part)
        else:
            processer = iter(Processer(
                filepath=dataset_path,
                part=part
            ))
            lsh = self.__train_trigram(processer,file_example=config.FILE_TEST,part=part)

        file = config.path_models +"_" + part + "_" + config.kGRAM
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
            if query_norm == None:
                return {'query': query, 'data': [], 'time': '0 ms', 'max': N, 'time_search': '0 ms',
                        'threshold': threshold}
            else:
                tokens = query_norm.split()
                tokens = [tokens[0] + " " + tokens[1], tokens[1] + " " + tokens[2]]

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
        return {'query': query, 'data': res_json, 'time': timing, 'max':N, 'time_search':timing_search, 'threshold':threshold}


def train_all(model):
    type = ['trigram', 'paragraph', 'section', 'phrase']
    for t in type:
        model.train(config.filepath, t)
        import gc
        gc.collect()
    exit()

def testing(model, all=True,type=''):
    if all:
        types = ['trigram', 'paragraph', 'section', 'phrase']
    else:
        types = [type]

    for t in types:
        empty = 0
        T = False
        if t == 'trigram':
            T = True
        type = t
        print("== {} == ".format(t))
        model.load_lsh("./model/model_" + type + "_"+config.kGRAM)
        print("model load!")

        if t == 'trigram':
            # per prendere le frasi come spunto per i trigrammi
            type = 'phrase'

        with open('test_' + type, 'rb') as handle:
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

        import gc
        gc.collect()

    exit()

if __name__ == '__main__':
    model = LSH()

    # ===== TRAIN ALL ======================================
    # train_all(model)

    # ===== PRINT TEST FILE .pickle ========================
    # type = 'section'
    # with open('test_' + type, 'rb') as handle:
    #     l = pickle.load(handle)
    # [ print(i) for i in l]
    # print("TOTAL: {}".format(len(l)))
    # exit()

    # ===== TESTING ========================================
    testing(model, all=False,type='phrase')

    # ===== TRAIN ==========================================
    # config.DEBUG = True
    type = 'trigram'
    model.train(config.filepath, type)
    import gc
    gc.collect()
    exit()



    # ===== SINGLE TEST =====================================
    # t = "trigram"
    # query = "This Decision will be applicable from this date of publication of the Commission Recommendation"
    # query ="""in addition, the commission will consult member states, the stakeholders and the authority
    #        to discuss the possibility to reduce the current maximum limits in all
    #        meat products and to further simplify the rules for the traditionally manufactured products"""
    # T = False
    # if t == 'trigram':
    #     T = True
    # model.load_lsh("./model/model_" + t + "_" + config.kGRAM)
    # res = model.predict(query, Trigram=T)
    # print("Q:", query)
    # print(json.dumps(res, ensure_ascii=False, indent=4))
    # exit()


