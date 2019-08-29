import socket

threshold_default = {
    'Section': 0.3,
    'Phrase': 0.80,
    'Paragraph': 0.50,
    "TriGram": 0.90
}
FILE_TEST = True
item_on_debug = 100
DEBUG = True
size_nlp = "sm"

permutations = 128
num_recommendations = 5
default_threshold = 0.0 #all
METRICS = "lev"


date_pattern = "(\d{2}|\d{1})(\s{1}|-|/)"+\
"((Jan(uary)?|(Feb(ruary)?|Ma(r(ch)?|y)|Apr(il)?|Jun(e)?|Jul(y)?|Aug(ust)|(Sept|Nov|Dec)(ember)?)|Oct(ober)?)|(\d{1}|\d{2}))"+\
               "(\s{1}|-|/)(\d{4}|(')?\d{2})"

abbr_dict = {
    'CEN': 'European Committee for Standardisation',
    'EEC': 'European Economic Commision',
    'EU': 'European Union',
    'EC': 'European Commission',
    'NATO': 'North Atlantic Treaty Organization',
    'USA': 'United States of America',
    'UK': 'United Kingdom',
    'PGI': 'Principal Global Indicators',
    'PDO': 'Protected designation of origin',
    'EFSA': 'European Food Safety Authority'
}
abbr_expand = "|".join(list(abbr_dict.keys()))

ip = socket.gethostbyname(socket.gethostname())

wordBased = True

if '130.136.4' in ip:
    filepath = './'
    path_models = '/public/antonio_conteduca/model_LSH/model'
else:
    filepath = '/home/anto/Scrivania/Tesi/dataset_train/'
    path_models = "/home/anto/Scrivania/Tesi/LSH/model/model"

print(">>>> RUN ON " + ip)
