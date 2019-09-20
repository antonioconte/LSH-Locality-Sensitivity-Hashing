import spacy

# 0. TEXT
# 1. remove_special_pattern
# 2. Detect Entity (COUNTRIES)
# 3. TOKENIZATION:
#     - marker 1,2 con < ... >
#     - rm punteggiatura e stopword + lemmatization
#     - marker <NUM>
# 4. N-GRAM Gen

import re
import config

class TextPipeline:
    def __init__(self,nlp,lang='english'):
        self.nlp = nlp
        try:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words(lang))
        except:
            import nltk
            nltk.download('stopwords')

    def generate_ngrams(self, tokens):
        k = int(config.kGRAM)
        tokens = [" ".join(tokens[i:i + k]).lower() for i in range(len(tokens) - k + 1)]
        return tokens

    def expand_abbr(self, text):
        text = re.sub(r'\(('+config.abbr_expand+')\)', '', text) #rimuove gli eventuali abbr_dict in parentesi
        for a in list(config.abbr_dict.keys()):
            reg = r'\b{}\b'.format(str(a))
            text = re.sub(reg, config.abbr_dict[a], text)
        return text

    def remove_special_pattern(self,text):
        pattern = {
            config.date_pattern: "DATE",
            "\(\d+\)+": 'NUMPAR',         #(NUM)
            "(\(|\s{1})\d+\.(\)|\s{1})": '',
            "\d+/\d+": 'NUMSLASH',  #NUM/NUM
            "(\s{1})?\d+\.\d+(\s{1})?": 'NUM'       #NUM.NUM
        }

        marker_list = []
        for key in pattern.keys():
            if pattern[key] == "DATE":
                text = re.sub(key,pattern[key],text,flags=re.IGNORECASE)
            else:
                text = re.sub(key,pattern[key],text)

            if pattern[key] != "":
                marker_list.append(pattern[key])
        return text,marker_list

    def norm_trigram(self,text):
        original = text
        edited = text
        tokens = text.split()

        for token in tokens:
            if 'DATE' in token:
                edited = edited.replace(token,"<date>")
                original_data = " ".join(token.split("DATE_")[1].split("_"))
                original = original.replace(token,original_data)
            if 'NUMSLASH' in token:
                edited = edited.replace(token, "<numslash>")
            elif token in self.stopwords:
                edited = edited.replace(token, "")
            edited = self.expand_abbr(edited)

        edited = " ".join(edited.lower().replace(","," ").replace("."," ").split())


        return original, edited

    def convert_trigram(self,text,Train=True):
        text = mark_date(text)
        text,_ = self.remove_special_pattern(text)

        tokens = text.split()
        if Train:
            trigrams = {}
        else:
            trigrams = []

        for i in range(len(tokens)):
            current_trigrams = ""
            k = 0
            pos_current = i
            while k < 3:
                # se non è una stopword
                if not tokens[pos_current] in self.stopwords:
                    current_trigrams += " " + tokens[pos_current]
                    k += 1
                else:
                    current_trigrams += " " + tokens[pos_current]

                if k == 3:
                    text_trigram, normalized_trigram = self.norm_trigram(current_trigrams.strip())
                    if Train:
                        trigrams[text_trigram] = [normalized_trigram]
                    else:
                        trigrams += [{text_trigram: [normalized_trigram]}]
                if pos_current >= len(tokens) -1 :
                    break

                pos_current += 1

        # import json
        # print(json.dumps(trigrams,indent=4,sort_keys=False))
        # exit()


        return trigrams

    def norm_text_trigram(self,query):
        ''' prende l'ultimo trigramma della stringa

        se la query non ha trigrami allora restituisce None che indica l'assenza di trigrammi
        '''
        # print(original, normalized[0]
        try:
            text = self.convert_trigram(query, Train=False)[-1]
            original = list(text.keys())[0]
            normalized = list(text.values())[0][0]
            return original,normalized
        except:
            return query,None

    def convert(self,text,divNGram=True):

        if len(text) < 5:
            if divNGram:
                return [""]
            else:
                return ""

        text = " ".join(text.split())  #rm spazi extra
        (text, special_pattern_list) = self.remove_special_pattern(text)

        list_Ent = ['COUNTRY']
        text = re.sub(config.countries_patt, 'COUNTRY', text)

        text = self.expand_abbr(text)
        doc = self.nlp(text)
        words = []

        list_sost = list_Ent + special_pattern_list
        #Part-of-speech tagging: https://spacy.io/usage/linguistic-features#pos-tagging
        for token in doc:
            if token.text in list_sost:
                words.append("<"+token.text+">")
            elif not token.is_stop and token.is_alpha: #is_alpha per rimuove anche la punteggiatura
                words.append(token.lemma_.lower())
            elif token.lemma_.isnumeric():
                words.append("<NUM>")

        if divNGram:
             return self.generate_ngrams(words)
        else:
            return " ".join(words)


def get_list_date(text, result=[]):
    date = re.search(config.date_pattern, text, flags=re.IGNORECASE)
    if not date:
        return result
    result += [date.group(0)]
    end = date.end()
    text = text[end:]
    return list(set(get_list_date(text, result)))

def mark_date(text):
    date_list = get_list_date(text)
    if len(date_list) == 0:
        return text
    for d in date_list:
        text = re.sub(d, "DATE_" + "_".join(d.split()), text)
    return text



if __name__ == '__main__':

    nlp = spacy.load('en_core_web_'+config.size_nlp)
    sample = """(25) Italy Germany the general and specific chemical requirements laid down by this directive 
    should aim at protecting the health of children from certain substances in toys, 
    while the 18 environmental concerns presented by toys are addressed by 
    horizontal environmental legislation applying to electrical and electronic toys, 
    namely directive 2002/95/EC of the european parliament and of 
    the council of 27 january 2003 on the restriction of the use of certain hazardous
    substances in electrical and electronic equipment and directive 2002/96/EC of 
    the european parliament and of the council of 27 january 2003 on waste electrical and electronic 
    equipment. in addition, environmental issues on waste are regulated by directive 2006/12/EC 
    of the european parliament and of the council of 5 april 2006, those on packaging and 
    packaging waste by directive 94/62/EC of the european parliament and of the council 
    of 20 december 1994 and those on batteries and accumulators and waste batteries 
    and accumulators by directive 2006/66/EC of the european parliament and of the council of 6 september 2006."""
    # config.kGRAM = 1
    # sample = "in addition, the commission will consult member states, the stakeholders and the authority to discuss the possibility to reduce the current maximum limits in all meat products and to further simplify the rules for the traditionally manufactured products"
    print("ORIGINAL: {}".format(sample))
    pip = TextPipeline(nlp)
    res = pip.convert(sample,divNGram=True)
    # res = pip.convert_trigram(sample)
    # res = pip.norm_text_trigram("hello world")
    # print(res[-1])

    import json
    print("\nEDITED: {}".format(json.dumps(res,indent=4)))

    # print(list(res.keys())[-1])