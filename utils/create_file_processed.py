import config
from preprocess import process_data
import pickle

types = ['phrase','section','paragraph']
ks = ['1','2','3']

import itertools
comb = list(itertools.product(types,ks))

for type,k in comb:
    p = process_data.Processer(filepath=config.filepath, part=type)
    data = p.run()
    with open('/home/anto/Scrivania/Tesi/testing/processed_data/' + type + '_' + k , 'wb') as f:
        pickle.dump(data, f)


# import json
# with open('/home/anto/Scrivania/Tesi/testing/processed_data/' + type + '_' + k + 'aaaaa', 'rb') as f:
#     data = pickle.load(f)
# print(json.dumps(data,indent=4))



