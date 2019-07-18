DEBUG = False
size_nlp = "sm"

permutations = 128
num_recommendations = 5

# METRICS = "jac"
# METRICS = "lev_sim"
METRICS = "lev"

filepath = '/home/anto/Scrivania/Tesi/dataset_train/'
# path_models = '/public/antonio_conteduca/model_LSH/model'
path_models = "/home/anto/Scrivania/Tesi/LSH/model/model"

date_pattern = "(\d{2}|\d{1})(\s{1}|-|/)"+\
"((Jan(uary)?|(Feb(ruary)?|Ma(r(ch)?|y)|Apr|Jun(e)?|Jul(y)?|Aug(ust)|(Sept|Nov|Dec)(ember)?)|Oct(ober)?)|(\d{1}|\d{2}))"+\
               "(\s{1}|-|/)(\d{4}|(')?\d{2})"
