'''
    input:
        filepath: directory contenente i documenti da normalizzare
        part    : Frase | Sezione | N-Gramma | Paragrafo


    output: {
        tag: [nomefile.html#Part-#numSeq] Text-not-Processed,
        data: [k-shingles di Text-Processed]
    }
'''
# html_txt = open(path + file, 'r', encoding='utf-8').read()
# soup = BeautifulSoup(html_txt, 'html.parser')

from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
import spacy
from tqdm import tqdm
import re
from preprocess.text_pipeline import TextPipeline

class Processer():
    def __init__(self,filepath="",part=""):
        self.nlp = spacy.load('en_core_web_sm')
        self.normalizer = TextPipeline(self.nlp)
        self.filepath = filepath
        self.part = part
        self.files = [f for f in listdir(self.filepath) if isfile(join(self.filepath, f))]
        print("Numero di documenti in",self.filepath," --->", len(self.files))

    def proc_paragraph(self, doc):
        par = doc.find_all("p")
        par_list = []
        res = []
        for p in par:
            txt = p.getText().strip()
            #process testo del paragrafo corrent
        return res

    def proc_phrase(self,filename,doc):
        par = doc.find_all("p")
        phrase_list = []
        for p in par:
            txt = p.getText().strip()
            if len(txt.split()) > 3:
                for sent in self.nlp(txt).sents:
                    phrase_list.append(sent)
        res = []
        for i,phrase in enumerate(phrase_list):
            string_phrase = str(phrase)
            data_list_normalized =  self.normalizer.convert(string_phrase)

            if len(data_list_normalized)>0:
                res.append({
                   'tag': '[' + filename + '#F' + str(i) + ']' + string_phrase,
                   'data': data_list_normalized
                })
        return res

    # def run_parallel(self):


    def run(self):
        if self.part == 'Frase':
            fun =  self.proc_phrase

        res = []

        for doc in tqdm(self.files,desc="Elaboring {}..".format(self.part)):
            html_txt = open(self.filepath + doc, 'r', encoding='utf-8').read()
            soup = BeautifulSoup(html_txt, 'html.parser')
            _ = [title.extract() for title in soup('p', {'class': 'doc-ti'})]
            res += fun(doc,soup)
        return res

if __name__ == '__main__':
    # DEBUG
    filepath = '/home/anto/Scrivania/Tesi/dataset/dataset_splitted/train/'
    part = "Frase"

    print("Processing docs...")
    processer = Processer(
        filepath = filepath,
        part = part
    )
    a = processer.run()
    # for i,frase in enumerate(a[0]):
    #     print(i,frase)
    import json

    print(json.dumps(a, indent=4, sort_keys=True))

