import json
import spacy
import LSH
from preprocess.text_pipeline import TextPipeline
import config


# ~~~~~ TRAIN ~~~~~~~~
def train(part):
    permutations = config.permutations
    # --------------------- FASE 1 -----------------------------#
    # DATO IL PATH CONTENENTE I DOCUMENTI
    # SI SCEGLIE IL TIPO DI PARTE INTERESSATA
    # type := section | paragraph | phrase | triGramm
    from preprocess.process_data import Processer
    processer = Processer(
        filepath = config.filepath,
        part=part
    )
    # Generazione del formato atteso da LSH.train
    data = processer.run()
    print(json.dumps(data, indent=4, sort_keys=True))
    #-----------------------------------------------------------#

    # --------------------- FASE 2 -----------------------------#
    lsh = LSH.train(data, permutations)
    # SAVING
    LSH.save_lsh(lsh,"./model/model_"+ part)


# ~~~~~~~ TEST ~~~~~~~~
def test(query,type):
    num_recommendations = config.num_recommendations
    permutations = config.permutations

    nlp = spacy.load('en_core_web_sm')
    normalizer = TextPipeline(nlp)
    lsh = LSH.load_lsh("../model/model_"+ type)
    import time
    from preprocess import utils
    query = utils.cleanhtml(query)
    start_time = time.time()
    result = LSH.predict(query, permutations, num_recommendations, lsh,normalizer)
    print(json.dumps(result, indent=4, sort_keys=True))
    print("Total Time: %.2f ms" % ((time.time() - start_time) * 1000))

if __name__ == '__main__':
    # type = "paragraph"
    type = "phrase"
    # type = "section"

    # train(type)
    # exit(1)


    query = """<p>The opportunities for establishing economic growth through innovation and a sustainable competitive energy policy have been recognised</p>"""
    test(query,type)


