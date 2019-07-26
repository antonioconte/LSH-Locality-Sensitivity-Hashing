import socket

DEBUG = True
size_nlp = "sm"

wordBased = False
permutations = 128
num_recommendations = 5
default_threshold = 0.0 #all
METRICS = "lev"

filepath = '/home/anto/Scrivania/Tesi/dataset_train/'

date_pattern = "(\d{2}|\d{1})(\s{1}|-|/)"+\
"((Jan(uary)?|(Feb(ruary)?|Ma(r(ch)?|y)|Apr|Jun(e)?|Jul(y)?|Aug(ust)|(Sept|Nov|Dec)(ember)?)|Oct(ober)?)|(\d{1}|\d{2}))"+\
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
print(ip)
if '130.136.4.222' in ip:
	path_models = '/public/antonio_conteduca/model_LSH/model'
if '130.136.4.145' in ip:
    path_models = '/public/antonio_conteduca/model_LSH_char/model'
else:
	path_models = "/home/anto/Scrivania/Tesi/LSH/model/model"
