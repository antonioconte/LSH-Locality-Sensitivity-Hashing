import json
import process_data
import pickle

# path_train = '/home/anto/Scrivania/Tesi/dataset_train/'
path_train = '/home/web/site181928/html/json_train/'

# path_save = '/home/anto/Scrivania/Tesi/testing/processed_data/trigrams/trigram_'
path_save = '/public/antonio_conteduca/processed_data/trigrams/trigram_'

# ========= TRIGRAM ============================
processer = iter(process_data.Processer(path_train, 'trigram'))
data = []
count = 0
p = 1
for item in processer:
    i = item[0]
    data += [i]
    count += 1
    if count != 0 and count % 50000 == 0:
        print("Scrivendo trigram_"+str(p))
        with open(path_save+str(p), 'wb') as f:
            pickle.dump(data, f)
        import gc
        gc.collect()
        data = []
        p += 1

print("Scrivendo ULTIMO trigram_"+str(p))
with open(path_save+str(p), 'wb') as f:
    pickle.dump(data, f)
import gc
gc.collect()
exit()
# ========== WRITE =============================
# with open('/home/anto/Scrivania/Tesi/testing/processed_data/trigrams/' , 'wb') as f:
#     pickle.dump(data, f)

# ========== READ =============================
# with open('/home/anto/Scrivania/Tesi/testing/processed_data/trigrams/trigram_1' , 'rb') as f:
#     data = pickle.load(f)


# ======== UNIT TRIGRAM FILE ===================
last = 19
data = []
for i in range(1,last+1):
    with open('/home/anto/Scrivania/Tesi/testing/processed_data/trigrams/trigram_' + str(i), 'rb') as f:
        data += [i for i in pickle.load(f)]

print(json.dumps(data,indent=4))
print(len(data))

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


