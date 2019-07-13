import textdistance

def metric(query, doc, normalizer, m=""):
    (tag,text) = doc.split("]")
    text_norm = " ".join(normalizer.convert(text, False))
    query_norm = " ".join(normalizer.convert(query, False))

    # rm num da query e da text
    if m == "jac":
        jac = textdistance.Jaccard()
        value = "%.2f" % jac(query_norm,text_norm)
    elif m == "lev":
        lev = textdistance.Levenshtein()
        value = int(lev.distance(query_norm,text_norm))
    elif m == "lev_sim":
        lev = textdistance.Levenshtein()
        # value = float(lev.distance(query_norm,text_norm))/float(len(query))
        value = lev.normalized_similarity(query_norm, text_norm)
        value = round(value, 2)
    else:
        value = 'NaN'

    return {'docname': tag[1:].split("#")[0],'text': text, m: value}
    # return {'text': tag, m: value}
