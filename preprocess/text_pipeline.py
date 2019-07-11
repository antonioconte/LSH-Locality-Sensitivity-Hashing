import spacy

# TEXT
# TOKENIZATION
# NORMALIZATION
# WORD LIST

class TextPipeline:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def convert(self,text):
        text = " ".join(text.split())
        doc = self.nlp(text)
        words = []
        for token in doc:
            if not token.is_stop and token.is_alpha:
                # print("\t",token.text,token.lemma_,token.pos_,)
                words.append(token.lemma_.lower())
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