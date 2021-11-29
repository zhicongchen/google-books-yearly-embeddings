import pandas as pd
import numpy as np
import logging
import os
import multiprocessing
import random
import gensim
import sys
from glob import glob
from tqdm import tqdm
from time import time
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
        self.count = 0
 
    def __iter__(self):
        for line in open(self.filename, "r"):            
            gram_str, match_count, volume_count = line.strip("\n").split("\t")
            gram_list = [i.split("_")[0] for i in gram_str.split(" ")]
            match_count = int(match_count)
            volume_count = int(volume_count)
            for i in np.arange(match_count):
                yield gram_list
                # if self.count <= 36000000:
                #     if np.random.random(1) > 0.5:
                #         yield gram_list
                #         self.count += 1


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 1
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        logging.info('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


def trim_function(word, count, min_count):
    if word in top_100000_frequent_words:
        return gensim.utils.RULE_KEEP
    else:
        return gensim.utils.RULE_DISCARD


n_cores = multiprocessing.cpu_count() # Count the number of cores in a computer
top_100000_frequent_words_df = pd.read_csv("top_100000_frequent_words.csv")
top_100000_frequent_words = top_100000_frequent_words_df["word"].values

# filename = "/data/zhicong/rawdata/google-ngram-v3-chi-sim/5grams_merge_by_decade/1990s.txt"
# filename = "./5grams_merge_by_year/2019.txt"
# year = filename.split("/")[-1].split(".")[0]

# print(sys.argv)
year = sys.argv[1].strip("\r")
print(year)

filename = "./5grams_merge_by_year/%s.txt" % year

# logfilename = "word2vec_training_tmp.log"
logfilename = "word2vec_training_%s_trim_iter_5.log" % year
if os.path.exists(logfilename):
    os.remove(logfilename)

logging.basicConfig(
    filename=logfilename, 
    format="%(asctime)s:%(levelname)s:%(message)s", 
    datefmt= '%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

model = Word2Vec(
    vector_size=300,         
    window=4, 
    min_count=5, 
    sg=1, 
    hs=0, 
    negative=5,
    ns_exponent=0.75,
    sample=1e-5,
    epochs=5,
    workers=n_cores-1
)

# write the parameters into the log file
logging.info("Hyperparamters: " + str(model.__dict__))

sentences = MySentences(filename) # a memory-friendly iterator
print("Building Vocabulary ...")
t = time()
# model.build_vocab(sentences, progress_per=1000000)
model.build_vocab(sentences, trim_rule=trim_function, progress_per=1000000)
logging.info('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

print("Training ...")
t = time()
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs, compute_loss=True, callbacks=[callback()])
logging.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# model.save("word2vec_tmp.model")
model.save("word2vec_%s.model" % year)
print("Model saved!")

