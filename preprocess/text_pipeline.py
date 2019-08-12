import spacy

# 0. TEXT
# 1. remove_special_pattern
# 2. Rilevo Soggetto (Dep Parsing)
# 3. Detect Entity (GPE)
# 4. TOKENIZATION:
#     - marker 2,3 con < ... >
#     - rm punteggiatura e stopword + lemmatization
#     - marker <num>
# 5. N-GRAM Gen

import re
import config


class TextPipeline:
    def __init__(self,nlp):
        self.nlp = nlp

    def generate_ngrams(self, tokens, k=3, word_based=True):
        # if len(s) < 5:
        #     n = len(s)
        # Replace all none alphanumeric characters with spaces

        if word_based:
            tokens = [" ".join(tokens[i:i + k]).lower() for i in range(len(tokens) - k + 1)]
        else:
            tokens = " ".join(tokens)
            k = 10
            tokens = [tokens[i:i + k] for i in range(len(tokens) - k + 1)]

        return tokens

    def remove_special_pattern(self,text):
        pattern = {
            config.date_pattern: "DATE",
            "\(\d+\)+": '',         #(NUM)
            "\d+/\d+": 'NUMSLASH',  #NUM/NUM
            "\d+\.\d+": 'NUM'       #NUM.NUM
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

    def convert(self,text,divNGram=True,wordBased=True):
        # print("> ", text)
        if len(text) < 5:
            return []
        text = " ".join(text.split())  #rm spazi extra
        (text, special_pattern_list) = self.remove_special_pattern(text)
        doc = self.nlp(text)

        # Dependency Parsing: https://spacy.io/api/annotation#dependency-parsing
        list_DPars = ['nsubj']
        for chunk in doc.noun_chunks:
            text_current = chunk.text
            try:
                if chunk.root.dep_ in list_DPars:
                    text = re.sub(r'\b{}\b'.format(text_current), chunk.root.dep_, text)
            except:
                pass

        # Detect Entity: https://spacy.io/api/annotation#named-entities
        list_Ent = ['GPE']
        for ent in doc.ents:
            try:
                if ent.label_ in list_Ent:
                    text = re.sub(ent.text, ent.label_,text)
            except:
                pass

        doc = self.nlp(text)
        list_sost = list_DPars + list_Ent + special_pattern_list

        words = []

        #Part-of-speech tagging: https://spacy.io/usage/linguistic-features#pos-tagging
        for token in doc:
            if token.text in list_sost:
                words.append("<"+token.text+">")
            elif not token.is_stop and token.is_alpha: #is_alpha per rimuove anche la punteggiatura
                words.append(token.lemma_.lower())
            elif token.lemma_.isnumeric():
                words.append("<num>")

        if divNGram:
             return self.generate_ngrams(words,word_based=wordBased)
        else:
            return " ".join(words)



if __name__ == '__main__':

    nlp = spacy.load('en_core_web_'+config.size_nlp)
    sample = """Well, prince, so genoa and Lucca are now just family estates of the Buonapartes. 
    1. It's my favourite pizza!
    2. Hello world!
    But I warn you, if you don’t tell me that this means war, 
    if you still try to defend the infamies and horrors perpetrated 
    by that Antichrist—I really believe he is Antichrist—I will have nothing more 
    to do with you and you are no longer my friend, 
    no longer my ‘faithful slave,’ as you call yourself! 
    But how do you do? I see I have frightened you—sit down and tell the news."""
    print("ORIGINAL: {}".format(sample))
    pip = TextPipeline(nlp)
    # res = pip.generate_ngrams(sample,k=11,word_based=False)
    res = pip.convert(sample)
    print("\nEDITED: {}".format(res))