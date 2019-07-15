from preprocess import preprocessing
import json
import spacy
import LSH
from preprocess.text_pipeline import TextPipeline
import config


# ~~~~~ TRAIN ~~~~~~~~
def train(type):
    permutations = config.permutations
    # --------------------- FASE 1 -----------------------------#
    # DATO IL PATH CONTENENTE I DOCUMENTI
    # SI SCEGLIE IL TIPO DI PARTE INTERESSATA
    # type := Sezione | Paragrafo | Frase | N-Gramma
    from preprocess.process_data import Processer
    processer = Processer(
        filepath = config.filepath,
        part="Frase"
    )
    data = processer.run()
    print(data)
    print(json.dumps(data, indent=4, sort_keys=True))
    print(len(data))
    #-----------------------------------------------------------#

    # --------------------- FASE 2 -----------------------------#
    lsh = LSH.train(data, permutations)
    # LSH.save_lsh(lsh,"./model/model"+ type)


# ~~~~~~~ TEST ~~~~~~~~
def test(query,type):
    num_recommendations = config.num_recommendations
    permutations = config.permutations

    nlp = spacy.load('en_core_web_sm')
    normalizer = TextPipeline(nlp)
    lsh = LSH.load_lsh("./model/model"+ type)
    import time
    start_time = time.time()
    result = LSH.predict(query, permutations, num_recommendations, lsh,normalizer)
    print(json.dumps(result, indent=4, sort_keys=True))
    timing = "Total Time: %.2f ms" % ((time.time() - start_time) * 1000)
    print(timing)

if __name__ == '__main__':
    # train("F")
    #
    # exit(1)
    query ="the Commission submitted the 1/07/19 a request to the Authority for further advice on a number of elements in relation to the submitted application"
    # query = """The measures provided for in this Regulation are in accordance with the opinion of the Standing Committee on the Food Chain and Animal Health and neither the European Parliament nor the Council have opposed them, The health claim set out in the Annex to this Regulation shall not be included in the Community list of permitted claims as provided for in Article 13(3) of Regulation (EC) This Decision will be applicable from this date of publication of the Commission Recommendation."""
    # query = """The measures provided for in this Regulation are in accordance with the opinion of the Standing Committee
    # on the Food Chain and Animal Health and neither the European Parliament nor the Council have opposed them,
    # The health claim set out in the Annex to this Regulation shall not be included in the Community
    # list of permitted claims as provided for in Article 1(23) of Regulation (EC)"""
    # query = """
    # Reporting requirement under Article 3 of the Euratom Treaty have been explain Commission Recommendation 2000-47455/Euratom.
    # """
    test(query,"F")




# ~~~~~ OLD VERSION ~~~~~~~~~~~~~~~~~~~~~~~~~~
# import LSH2 as LSH
# db = pd.read_csv('dataset/papers.csv')[:200]
# db['text'] = db['title'] + " " + db['abstract']
