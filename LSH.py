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

    def __train_trigram(self,data,file_example=False):
        start_time = time.time()
        forest = MinHashLSHForest(num_perm=config.permutations)
        num_trigram = 0
        example = []
        for item in tqdm(data):
            num_trigram += 1
            tokens = item[0]['data']
            tag = item[0]['tag']
            if random.randint(0, 100) == 0:
                item_choice = tag.split("]")[1]
                if len("".join(item_choice.split("]"))) > 8:
                    example += [ item_choice ]

            m = MinHash(num_perm=config.permutations)
            for s in tokens:
                m.update(s.encode('utf8'))
            forest.add(tag,m)
            next(data)

        if file_example:
            f = open("Test/trigram_test.json", 'w')
            f.write(json.dumps(example, indent=4, sort_keys=True, ensure_ascii=False))
            f.close()

        print("------> Total: " + str(num_trigram))
        forest.index()
        print('It took %.2f seconds to build forest.' % (time.time() - start_time))

        return forest


    def __train(self,data,type,file_example=False):
        start_time = time.time()
        forest = MinHashLSHForest(num_perm=config.permutations)
        example = []
        for item in tqdm(data, desc="MinHash Docs.."):
            tag = item['tag']
            tokens = item['data']
            import random
            if random.randint(0, 100) < 10:
                example += [tag.split("]",1)[1]]
                print(tag.split("]",1)[1])
            m = MinHash(num_perm=config.permutations)
            for s in tokens:
                m.update(s.encode('utf8'))
            forest.add(tag,m)

        forest.index()
        if file_example:
            f = open("Test/"+type+"_test.json", 'w')
            f.write(json.dumps(example, indent=4, sort_keys=True, ensure_ascii=False))
            f.close()
        print('It took %.2f seconds to build forest.' % (time.time() - start_time))

        return forest


    def train(self,dataset_path,part):
        print("Training {}...".format(part))
        from preprocess.process_data import Processer
        if not part == "trigram":
            processer = Processer(
                filepath=dataset_path,
                part=part
            )
            data = processer.run()
            lsh = self.__train(data,part,file_example=config.FILE_TEST)
        else:
            processer = iter(Processer(
                filepath=dataset_path,
                part=part
            ))
            lsh = self.__train_trigram(processer,file_example=config.FILE_TEST)
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


def predict(type):
    lsh.load_lsh("./model/model_" + type)
    query = "(25) the general and specific chemical requirements laid down by this directive should aim at protecting the health of children from certain substances in toys, while the environmental concerns presented by toys are addressed by horizontal environmental legislation applying to electrical and electronic toys, namely directive 2002/95/european commission of the european parliament and of the council of 27 january 2003 on the restriction of the use of certain hazardous substances in electrical and electronic equipment and directive 2002/96/european commission of the european parliament and of the council of 27 january 2003 on waste electrical and electronic equipment. in addition, environmental issues on waste are regulated by directive 2006/12/european commission of the european parliament and of the council of 5 april 2006, those on packaging and packaging waste by directive 94/62/european commission of the european parliament and of the council of 20 december 1994 and those on batteries and accumulators and waste batteries and accumulators by directive 2006/66/EC of the european parliament and of the council of 6 september 2006."
    # query = "provide conformity assessment"
    # query = """21 october 2006"""

    T = False
    if type == "trigram":
        T = True

    res = lsh.predict(query,Trigram=T)
    print(json.dumps(res, ensure_ascii=False, indent=4))
    exit(1)

if __name__ == '__main__':
    lsh = LSH()

    config.DEBUG = False
    # predict("trigram")
    # model_type_train = ["paragraph", "section","phrase","trigram"]
    # model_type_train = ["paragraph", "section","phrase"]
    model_type_train = ["trigram"]
    for m in model_type_train:
        lsh.train(config.filepath, m)
        import gc
        gc.collect()
