import distance

def metric(query_norm, doc, normalizer, m=""):
    '''
    :param query: prima stringa di confronto
    :param doc: seconda stringa di confronto
    :param normalizer: oggetto per normalizzare testo (pipeline)
    :return: { nomedocumento, testo con cui ha valore di similarità, similarità metrica }
    '''
    (tag,text) = doc.split("]")
    text_norm = normalizer.convert(text, False)
    # print("QUERY", query_norm)
    # print("RES", text_norm)
    value = "%.2f" % (1.0 - distance.nlevenshtein(query_norm, text_norm,method=1))
    return {'docname': tag[1:].split("#")[0],'text': text, 'lev': value}
    # return {'text': tag, m: value}

