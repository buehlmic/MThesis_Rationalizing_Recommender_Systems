import numpy as np
import pandas as pd
from data_sents_5 import *
from data_random_samps import *
from data_indep_sents import *

def load_data():
    #data = data_fixed + data_adaptive + data_const_context + data_train_context + \
    data = []
    data = data_5 + data_random_sents + data_indep_sents
    frames = []
    
    for d in data:
      F_pos = [x[1] for x in d[0]]
      F_neg = [x[2] for x in d[0]]
      F_score = [x[0] for x in d[0]]
      F = np.vstack([F_score, F_pos, F_neg])
      F = pd.DataFrame(F, index=['s', 'p', 'n']).T
      F['data'] = d[1]
      F['num_sents'] = d[2]
      F['random_samp'] = d[3]
      F['context'] = d[4]  
      F['adaptive_lrs'] = d[5]
      frames.append(F)
    
    return pd.concat(frames, ignore_index=True)
