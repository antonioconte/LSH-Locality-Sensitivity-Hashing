from preprocess import preprocessing
import json
import spacy


permutations = 128
num_recommendations = 5

nlp = spacy.load('en_core_web_sm')
filepath = 'dataset_html/'
data = preprocessing.processing_data(filepath,nlp,"F")

# print(json.dumps(data, indent=4, sort_keys=True))
# print(len(data))
# exit(1)

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

