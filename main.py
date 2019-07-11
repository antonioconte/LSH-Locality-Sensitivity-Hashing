from preprocess import preprocessing
import json
import spacy
import LSH
from preprocess.text_pipeline import TextPipeline

permutations = 128
num_recommendations = 5

# ~~~~~~ TRAIN ~~~~~~~~
def train(type):
    # --------------------- FASE 1 -----------------------------#
    # DATO IL PATH CONTENENTE I DOCUMENTI
    # SI SCEGLIE IL TIPO DI PARTE INTERESSATA
    # type := Sezione | Paragrafo | Frase | N-Gramma

    print("Load Spacy...")
    nlp = spacy.load('en_core_web_sm')
    normalizer = TextPipeline(nlp)

    filepath = '/home/anto/Scrivania/Tesi/dataset/dataset_splitted/train/'
    data = preprocessing.processing_data(filepath,nlp,"F",normalizer)

    # print(json.dumps(data, indent=4, sort_keys=True))
    # print(len(data))
    # exit(1)
    #------------------------------------------------------------#

    # --------------------- FASE 2 -----------------------------#
    lsh = LSH.train(data, permutations)
    LSH.save_lsh(lsh,"./model/model"+ type)


# ~~~~~~~ TEST ~~~~~~~~
def test(text,type):
    nlp = spacy.load('en_core_web_sm')
    normalizer = TextPipeline(nlp)
    lsh = LSH.load_lsh("./model/model"+ type)
    result = LSH.predict(title, permutations, num_recommendations, lsh,normalizer)
    print(json.dumps(result, indent=4, sort_keys=True))

title = """On operator in question is Le Seysselan GAEC, Vallod, SEYSSEL."""
# title = """this is a european format 2332"""

test(title,"F")
# train("F")




# ~~~~~ OLD VERSION ~~~~~~~~~~~~~~~~~~~~~~~~~~
# import LSH2 as LSH
# db = pd.read_csv('dataset/papers.csv')[:200]
# db['text'] = db['title'] + " " + db['abstract']
