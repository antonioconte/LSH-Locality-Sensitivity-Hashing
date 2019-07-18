DEBUG = False


permutations = 128
num_recommendations = 5

# METRICS = "jac"
# METRICS = "lev_sim"
METRICS = "lev"

filepath = '/home/anto/Scrivania/Tesi/dataset_train/'




date_pattern = "(\d{2}|\d{1})(\s{1}|-|/)"+\
"((Jan(uary)?|(Feb(ruary)?|Ma(r(ch)?|y)|Apr|Jun(e)?|Jul(y)?|Aug(ust)|(Sept|Nov|Dec)(ember)?)|Oct(ober)?)|(\d{1}|\d{2}))"+\
               "(\s{1}|-|/)(\d{4}|(')?\d{2})"
