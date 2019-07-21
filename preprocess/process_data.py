'''
    input:
        filepath: directory contenente i documenti da normalizzare
        part    : Frase | Sezione | N-Gramma | Paragrafo


    output: [{
        tag: [nomefile.html#Part-#numSeq] Text-not-Processed,
        data: [k-shingles di Text-Processed]
    }]
'''
# html_txt = open(path + file, 'r', encoding='utf-8').read()
# soup = BeautifulSoup(html_txt, 'html.parser')

import spacy
from tqdm import tqdm
from preprocess.text_pipeline import TextPipeline
import json

class Processer():
    def __init__(self,filepath="",part=""):
        self.filepath = filepath + "total_" + part + ".json"
        self.nlp = spacy.load('en_core_web_sm')
        self.normalizer = TextPipeline(self.nlp)
        if part == "paragraph" or part == "section":
            self.tag = part[0].upper()
        else:
            self.tag = "F"

        print(self.filepath)
        with open(self.filepath) as json_file:
            self.data = json.load(json_file)
        print("TOTAL {}: {}".format(part,self.data['total']))

    def run(self):
        result = []
        # docList = list(self.data['data'].keys())[1:5]
        docList = list(self.data['data'].keys())
        for docname in tqdm(docList):
            items_of_doc = self.data['data'][docname]
            # print("doc {} ha {} {}".format(docname, len(items_of_doc),self.tag))
            for (i,item) in enumerate(items_of_doc):
                data_list_normalized = self.normalizer.convert(item)
                if len(data_list_normalized) > 0:
                    result += [
                        {
                            'tag': '[' + docname + '#' + self.tag + str(i) + ']' + item,
                            'data': data_list_normalized

                        }
                    ]
        return result

if __name__ == '__main__':
    # p = Processer('/home/anto/Scrivania/Tesi/dataset_train/', 'paragraph')
    # p = Processer('/home/anto/Scrivania/Tesi/dataset_train/', 'section')
    p = Processer('/home/anto/Scrivania/Tesi/dataset_train/', 'phrase')
    print(json.dumps(p.run(), indent=4, sort_keys=True))


