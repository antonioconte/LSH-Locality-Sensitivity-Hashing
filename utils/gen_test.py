import pickle
import json
import random

k = '3'

t = 'paragraph'
# t = 'section'
# t = 'phrase'

NUM_TEST = 40
with open('/home/anto/Scrivania/Tesi/testing/processed_data/' + t + '_' + k, 'rb') as handle:
    data = pickle.load(handle)

queries_test = []
idx = random.sample(range(len(data)), NUM_TEST)
for id in idx:
    item = data[id]['tag'].split("]", 1)[1]
    queries_test += [item]

with open('/home/anto/Scrivania/Tesi/testing/testing_file/'+t, 'wb') as handle:
    pickle.dump(queries_test, handle)

[ print(">", i) for i in queries_test]
print("#",NUM_TEST)
# print(json.dumps(data, ensure_ascii=False, indent=4))
