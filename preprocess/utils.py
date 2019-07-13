import re

def preprocess(text,k=3):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.lower()
    # tokens = tokens.removestopword()
    # tokens = stemming
    tokens = tokens.split()
    tokens = [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
    return tokens

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, ' ', raw_html)
  cleanr = re.compile('&nbsp;')
  cleantext = " ".join(re.sub(cleanr, '', cleantext).split())
  return cleantext


# sample = """
# <section class="selection">
# <p>Article 1</p><h4><strong>Union target&nbsp;</strong></h4><p class="normal">1. The Union target, as referred to in Article 4(1) of Regulation (EC) No 2160/2003, for the reduction of <span class="italic">Salmonella</span> Enteritidis and <span class="italic">Salmonella</span> Typhimurium in turkeys (‘Union target’) shall be:</p><table border="0" cellpadding="0" cellspacing="0" width="100%" class="mce-item-table"> <tbody><tr><td valign="top"><p class="normal">(a)</p></td><td valign="top"><p class="normal">a reduction of the maximum annual percentage of fattening turkey flocks remaining positive of <span class="italic">Salmonella</span> Enteritidis and <span class="italic">Salmonella</span> Typhimurium to 1 % or less; and</p></td></tr></tbody></table><table border="0" cellpadding="0" cellspacing="0" width="100%" class="mce-item-table"> <tbody><tr><td valign="top"><p class="normal">(b)</p></td><td valign="top"><p class="normal">a reduction of the maximum annual percentage of adult breeding turkey flocks remaining positive of <span class="italic">Salmonella</span> Enteritidis and <span class="italic">Salmonella</span> Typhimurium to 1 % or less.</p></td></tr></tbody></table><p class="normal">However, for Member States with less than 100 flocks of adult breeding or fattening turkeys, the Union target shall be that annually no more than one flock of adult breeding or fattening turkeys may remain positive.</p><p class="normal">As regards monophasic <span class="italic">Salmonella</span> Typhimurium, serotypes with the antigenic formula 1,4,[5],12:i:- shall be included in the Union target.</p><p class="normal">2. The testing scheme necessary to verify progress in the achievement of the Union target is set out in the Annex (‘testing scheme’).</p></section>
# """
# print1(cleanhtml(sample))