import config
from MinhashLSH import Minhash
import pickle
from tqdm import tqdm
import time
import random
import json

def train_all(k = '3'):
    # type = ['paragraph', 'section', 'phrase']
    # for t in type:
    #     model = Minhash(t, k=k)
    #     model.train(config.filepath)
    #     import gc
    #     gc.collect()

    model = Minhash('trigram',k='3')
    model.train()

if __name__ == '__main__':
    # ===== TRAIN ALL ======================================
    # config.DEBUG = True
     # k = { '1','2', '3'}

    # train_all(k='1')
    # train_all(k='2')
    train_all(k='3')