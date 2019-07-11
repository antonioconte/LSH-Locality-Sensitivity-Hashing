from os import listdir
from os.path import isfile, join
from preprocess import preprocessing
import json
import spacy
from tqdm import tqdm
permutations = 128
num_recommendations = 5

def processing_data(filepath, nlp, Type):
    # ~~~~~~ LOAD DATA ~~~~
    file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    # PROCESSING DATA
    data = [
        obj
        for file in tqdm(file_list,desc="Loading File from {}".format(filepath))
            for obj in preprocessing.process_doc(filepath, file, nlp, type=Type)
    ]
    return data


nlp = spacy.load('en_core_web_sm')
filepath = 'dataset_html/'
data = processing_data(filepath,nlp,"F")


print(json.dumps(data, indent=4, sort_keys=True))
# print(len(data))
exit(1)

# OLD VERSION
# import LSH2 as LSH
# db = pd.read_csv('dataset/papers.csv')[:200]
# db['text'] = db['title'] + " " + db['abstract']

# ~~~~~~ TRAIN ~~~~~~~~
import LSH
# lsh = LSH.train(data, permutations)
# LSH.save_lsh(lsh)
title = 'If a proposed Appendix is related to an amendment to the Convention or an Annex, proposed for adoption in accordance with Article 15 or Article 17, the proposal, adoption and entry into force of that Appendix shall be governed by the same provisions as apply to the proposal, adoption and entry into force of that amendment.'


# ~~~~~~~ TEST ~~~~~~~~
lsh = LSH.load_lsh("model")

result = LSH.predict(title, permutations, num_recommendations, lsh)
print(json.dumps(result, indent=4, sort_keys=True))

