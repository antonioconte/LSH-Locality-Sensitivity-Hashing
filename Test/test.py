from LSH import LSH
import spacy
from statistics import mean
import time

type = ['trigram','section', 'phrase', 'paragraph'][1:]

def testing(type):
    print("================= {} =================".format(type))
    import json
    filepath = "./Test/{}_test.json".format(type)
    with open(filepath) as json_file:
        data = json.load(json_file)
    print("JSON ({}) LOADED: {}".format(type,len(data)))
    time.sleep(1)
    Trigram = False
    if type == "trigram":
        Trigram = True

    lsh = LSH()
    lsh.load_lsh("./model/model_{}".format(type))
    print("LSH {} model: loaded!".format(type))

    search_time = []
    total_time = []
    mean_lev = []
    zeros = 0
    import tqdm
    for query in tqdm.tqdm(data):
        try:
            result = lsh.predict(query,N=10,Trigram=Trigram)
            search_time += [ float(result['time_search'].split(" ")[0]) ]
            total_time += [  float(result['time'].split(" ")[0]) ]
            if len(result['data']) > 0:
                current_res = [ float(item['lev']) for item in result['data']]
                mean_lev += [ mean(current_res) ]
            else:
                zeros += 1
                mean_lev += [ 0.0 ]
        except KeyboardInterrupt:
            print("Time (Search): {} ms".format(round(mean(search_time), 2)))
            print("Time (Total): {} ms".format(round(mean(total_time), 2)))
            print("Lev (Mean): {}".format(round(mean(mean_lev), 2)))
            print("#NoResult: {}".format(zeros))
            exit(1)
        except:
            print(query)
            pass
    # time.sleep(1)
    print("Time (Search): {} ms".format(round(mean(search_time),2)))
    # print(search_time)
    print("Time (Total): {} ms".format(round(mean(total_time),2)))
    # print(total_time)
    print("Lev (Mean): {}".format(round(mean(mean_lev),2)))
    # print(mean_lev)
    print("#NoResult: {}".format(zeros))

    print("================= END {} =============\n".format(type))


for t in type:
    testing(t)