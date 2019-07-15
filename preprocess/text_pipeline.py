import spacy

# TEXT
# TOKENIZATION
# NORMALIZATION
# WORD LIST

import re
import config

class TextPipeline:
    def __init__(self,nlp,size="sm"):
        self.size = size # lg
        self.nlp = nlp

    def generate_ngrams(self, tokens, k=3):
        # if len(s) < 5:
        #     n = len(s)
        # Replace all none alphanumeric characters with spaces
        # s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
        tokens = [" ".join(tokens[i:i + k]).lower() for i in range(len(tokens) - k + 1)]

        # Break sentence in the token, remove empty tokens
        # tokens = [token for token in s.split(" ") if token != ""]

        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        # ngrams = zip(*[tokens[i:] for i in range(n)])
        # return [" ".join(ngram) for ngram in ngrams]
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

    def convert(self,text,divNgram=True):
        text = " ".join(text.split())
        (text, special_pattern_list) = self.remove_special_pattern(text)
        doc = self.nlp(text)
        words = []

        # Dependency Parsing: https://spacy.io/api/annotation#dependency-parsing
        list_DPars = ['nsubj']
        for chunk in doc.noun_chunks:
            text_current = chunk.text
            try:
                if chunk.root.dep_ in list_DPars:
                    text = re.sub(text_current, chunk.root.dep_, text)
            except:
                print("SALTO_DPARS:",text_current)
                pass


        # Detect Entity: https://spacy.io/api/annotation#named-entities
        list_Ent = ['GPE']
        for ent in doc.ents:
            try:
                if ent.label_ in list_Ent:
                    text = re.sub(ent.text, ent.label_,text)
            except:
                print("SALTO_ENT:",ent.text)
                pass
        doc = self.nlp(text)
        list_sost = list_DPars + list_Ent + special_pattern_list

        #Part-of-speech tagging: https://spacy.io/usage/linguistic-features#pos-tagging
        for token in doc:
            if token.text in list_sost:
                words.append("<"+token.text+">")
            elif not token.is_stop and token.is_alpha: #is_alpha per rimuove anche la punteggiatura
                # print("\t",token.text,token.lemma_,token.pos_,)
                words.append(token.lemma_.lower())
            elif token.lemma_.isnumeric():
                words.append("<NUM>")

        if divNgram:
             return self.generate_ngrams(words)
        else:
            return " ".join(words)



if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    sample = """Well, Prince, so Genoa and Lucca are now just family estates of the Buonapartes. 
    1. It's my favourite pizza!
    2. Hello world!
    But I warn you, if you don’t tell me that this means war, 
    if you still try to defend the infamies and horrors perpetrated 
    by that Antichrist—I really believe he is Antichrist—I will have nothing more 
    to do with you and you are no longer my friend, 
    no longer my ‘faithful slave,’ as you call yourself! 
    But how do you do? I see I have frightened you—sit down and tell me all the news."""
    print("ORIGINAL: {}".format(sample))
    pip = TextPipeline(nlp)
    res = pip.convert(sample)
    print("\nEDITED: {}".format(res))