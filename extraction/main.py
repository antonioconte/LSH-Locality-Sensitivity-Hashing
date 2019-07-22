'''
    estrazione di frasi trigrammi paragrafi e sezioni
'''
from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import json
import re


def extract_phrase(filepath,test=False):
    files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    total = 0
    phrase_dict = {'data': {} }
    for doc in tqdm(files):
        html_txt = open(filepath + doc, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_txt, 'html.parser')
        _ = [title.extract() for title in soup('p', {'class': 'doc-ti'})]
        par = soup.find_all("p")
        text_par = ""
        for p in par:
            text_par += str(p)
        # split in .</p>
        text_phrase = re.sub(r'\.</p>', '.', text_par)
        # rm dei tag di mezzo
        text_phrase = re.sub(r'<[^>]*>', ' ', text_phrase)
        text_phrase = re.sub(r'Article [0-9]*', '', text_phrase)
        # lista testi fino al punto
        list_phrase_point = text_phrase.split(".")
        list_phrase = []
        for f in list_phrase_point:
            list_phrase += [" ".join(f.split()) for f in re.sub(r":|;",'ENDFRASE',f).split("ENDFRASE") if len(f.strip().split()) > 5]

        phrase_dict['data'][doc] = list_phrase
        total += len(list_phrase)
    # print(json.dumps(phrase_dict,indent=4,))
    if test:
        f = open("test_total_phrase.json",'w')
    else:
        f = open("total_phrase.json",'w')

    phrase_dict['total'] = total
    f.write(json.dumps(phrase_dict, indent=4, sort_keys=True,ensure_ascii=False))
    f.close()

def extract_paragraph(filepath,test=False):
    files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    total = 0
    par_dict =  {'data': {} }
    for doc in tqdm(files):
        html_txt = open(filepath + doc, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_txt, 'html.parser')
        _ = [title.extract() for title in soup('p', {'class': 'doc-ti'})]
        par = soup.find_all("p")
        text_par = ""
        for p in par:
            text_par += " " + str(p)
        # split in .</p>
        text_par_clear = re.sub(r'\.</p>', '.ENDPAR', text_par) # identifico la fine del paragrafo
        text_par_clear = re.sub(r'<[^>]*>', '', text_par_clear) # rimuove i tag html
        text_par_clear = re.sub(r'Article [0-9]*', '', text_par_clear)

        # lista testi fino al punto
        list_par = [" ".join(p.split()) for p in text_par_clear.split("ENDPAR") if len(p.strip().split()) > 5]
        par_dict['data'][doc] = list_par
        total += len(list_par)
        # print(json.dumps(par_dict, indent=4, sort_keys=True))

    # print(total)
    par_dict['total'] = total
    if test:
        f = open("test_total_paragraph.json",'w')
    else:
        f = open("total_paragraph.json",'w')

    f.write(json.dumps(par_dict, indent=4, sort_keys=True,ensure_ascii=False))
    f.close()


def extract_section(filepath,test=False):
    files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    total = 0
    sec_dict =  {'data': {} }
    for doc in tqdm(files):
        html_txt = open(filepath + doc, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_txt, 'html.parser')
        _ = [title.extract() for title in soup('p', {'class': 'doc-ti'})]
        sections = soup.find_all("section")
        sections_list = [ " ".join(re.sub(r'<[^>]*>', ' ', str(s)).split()) for s in sections]
        total += len(sections)
        sec_dict['data'][doc] = sections_list
        # print(json.dumps(par_dict, indent=4, sort_keys=True))

    # print(total)
    sec_dict['total'] = total
    if test:
        f = open("test_total_section.json",'w')
    else:
        f = open("total_section.json",'w')

    f.write(json.dumps(sec_dict, indent=4, sort_keys=True,ensure_ascii=False))
    f.close()

filepath = '/home/anto/Scrivania/Tesi/dataset/dataset_splitted/test/'
extract_phrase(filepath,True)
extract_paragraph(filepath,True)
extract_section(filepath,True)


filepath = '/home/anto/Scrivania/Tesi/dataset/dataset_splitted/train/'
extract_phrase(filepath)
extract_paragraph(filepath)
extract_section(filepath)