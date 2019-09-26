import json
import config
from preprocess import process_data
import pickle


# ========= TRIGRAM ============================
processer = iter(process_data.Processer('/home/anto/Scrivania/Tesi/dataset_train/', 'trigram'))
data = []
for item in processer:
    i = item[0]
    data += [i]
# ========== WRITE =============================
with open('/home/anto/Scrivania/Tesi/testing/processed_data/trigram' , 'wb') as f:
    pickle.dump(data, f)

# ========== READ =============================
# with open('/home/anto/Scrivania/Tesi/testing/processed_data/trigram' , 'rb') as f:
#     data = pickle.load(f)
# print(json.dumps(data,indent=4))


exit()




# ==============================================
types = ['phrase','section','paragraph']
ks = ['1','2','3']

import itertools
comb = list(itertools.product(types,ks))

# for type,k in comb:
#     p = process_data.Processer(filepath=config.filepath, part=type)
#     data = p.run()
#     with open('/home/anto/Scrivania/Tesi/testing/processed_data/' + type + '_' + k , 'wb') as f:
#         pickle.dump(data, f)

for type,k in comb:
    with open('/home/anto/Scrivania/Tesi/testing/processed_data/' + type + '_' + k, 'rb') as f:
        data = pickle.load(f)
    print("> {}_{}: {}".format(type,k,len(data)))

print(json.dumps(data,indent=4))


