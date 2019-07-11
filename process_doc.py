from bs4 import BeautifulSoup
import spacy
import LSH

# considero frasi con almeno 4 parole
def extract_phrase(filename,soup, lang="en",min_count=5):
    res = []
    text = []
    par = soup.find_all("p")
    for p in par:
        txt = p.getText().strip()
        if len(txt.split()) >= min_count:
            text.append(txt)
    text = " ".join(text)
    nlp = spacy.load(lang + '_core_web_sm')
    doc = nlp(text)
    num_f = 0

    for sent in doc.sents:
        sent = sent.text.strip()
        if len(sent.split()) >= min_count:
            num_f += 1
            res.append({
                'tag': '[' + filename + '#F' + str(num_f) + ']' + sent,
                'data': LSH.preprocess(sent)
            })
    return res

# type = Sezione-Paragrafo-Ngramma-Frase "SPNF"
def process_doc(path, file, type=""):
    html_txt = open(path + file, 'r', encoding='utf-8').read()
    soup = BeautifulSoup(html_txt, 'html.parser')
    _ = [script.extract() for script in soup('p', {'class': 'doc-ti'})]
    res = extract_phrase(file,soup)
    return res




def main():
    src_path = "/home/anto/Scrivania/Tesi/dataset/EN/"
    file = "21997A0319(01).html"
    import json
    res = process_doc(src_path,file,type="S")
    # res = process_docs(src_path,"F")
    print(json.dumps(res, indent=4, sort_keys=True))

if __name__ == '__main__':
    main()



