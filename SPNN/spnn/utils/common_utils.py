import random
import numpy as np
import torch as th

def set_seed_thread(seed,threadnum):
    th.set_num_threads(threadnum)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
