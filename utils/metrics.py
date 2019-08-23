import distance
import Levenshtein


def sim_lev(str1, str2):
    lev = Levenshtein.distance(str1, str2)
    return ( 1.0 - (lev/max(len(str1),len(str2))))

def metric(query_norm, doc, normalizer,Trigram = False):
    '''
    :param query: prima stringa di confronto
    :param doc: seconda stringa di confronto
    :param normalizer: oggetto per normalizzare testo (pipeline)
    :return: { nomedocumento, testo con cui ha valore di similarità, similarità metrica }
    '''
    (tag,text) = doc.split("]",1)
    if Trigram:
        _,text_norm = normalizer.norm_text_trigram(text)
    else:
        text_norm = normalizer.convert(text, False)

    # print("QUERY", query_norm)
    # print("RES", text_norm)

    value = "%.2f" % sim_lev(query_norm,text_norm) # FAST
    return {'docname': tag[1:].split("#")[0],'text': text, 'lev': value}


