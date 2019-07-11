import spacy

# TEXT
# TOKENIZATION
# NORMALIZATION
# WORD LIST

import re


class TextPipeline:
    def __init__(self,nlp,size="sm"):
        self.size = size # lg
        self.nlp = nlp

    def generate_ngrams(self, s, n=5):
        if len(s) < 5:
            n = len(s)

        s = " ".join(s)
        # Convert to lowercases
        s = s.lower()

        # Replace all none alphanumeric characters with spaces
        # s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

        # Break sentence in the token, remove empty tokens
        tokens = [token for token in s.split(" ") if token != ""]

        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def convert(self,text,divNgram=True):
        text = " ".join(text.split())
        doc = self.nlp(text)
        words = []
        # TODO: marcare soggetto qui con dependency parser
        for token in doc:
            if not token.is_stop and token.is_alpha:
                # print("\t",token.text,token.lemma_,token.pos_,)
                words.append(token.lemma_.lower())
            elif token.lemma_.isnumeric():
                words.append("<NUM>")
        if divNgram:
             return self.generate_ngrams(words)
        else:
            return words



if __name__ == '__main__':
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
    pip = TextPipeline()
    res = pip.convert(sample)
    print("\nEDITED: {}".format(res))