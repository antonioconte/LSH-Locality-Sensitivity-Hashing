from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from bs4 import BeautifulSoup
from preprocess import utils
import config

# considero frasi con almeno 5 parole
def extract_phrase(filename,soup,nlp,normalizer,min_count=5):
    res = []
    text = []
    par = soup.find_all("p")
    for p in par:
        txt = p.getText().strip()
        if len(txt.split()) >= min_count:
            text.append(txt)
    text = " ".join(text)
    doc = nlp(text)
    num_f = 0

    for sent in doc.sents:
        sent = sent.text.strip()
        if len(sent.split()) >= min_count:
            num_f += 1
            res.append({
                'tag': '[' + filename + '#F' + str(num_f) + ']' + sent,
                'data': normalizer.convert(sent) # corpus del minhashing
            })
    return res

# type = Sezione-Paragrafo-Ngramma-Frase "SPNF"
def process_doc(path, file, nlp,normalizer, type="F"):
    html_txt = open(path + file, 'r', encoding='utf-8').read()
    soup = BeautifulSoup(html_txt, 'html.parser')
    _ = [script.extract() for script in soup('p', {'class': 'doc-ti'})]
    res = {}

    if type == "F":
        res = extract_phrase(file,soup,nlp,normalizer)
    elif type == "S":
        res = {}
    elif type == "P":
        res = {}
    elif type == "N":
        res = {}
    return res

def processing_data(filepath, nlp, Type,normalizer):
    # ~~~~~~ LOAD DATA ~~~~
    if not config.DEBUG:
        file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    else:
        file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))][:10]

    # PROCESSING DATA
    data = [
        obj
        for file in tqdm(file_list,desc="Loading File from {}".format(filepath))
            for obj in process_doc(filepath, file, nlp, normalizer, type=Type)
    ]
    return data


def main():
    # filepath = config.
    res = {}
    import json
    print(json.dumps(res, indent=4, sort_keys=True))

if __name__ == '__main__':
    main()



