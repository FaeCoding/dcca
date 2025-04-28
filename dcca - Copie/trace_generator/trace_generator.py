import sys
from pathlib import Path
import pickle
# Get the parent directory path
parent_dir = str(Path(__file__).resolve().parent.parent)
# Add it to Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from requirements import *
from functions import *


nb_bit = 2048
wordsize = 32

nb_traces_to_generate = int(input("How much exponentiations do you want ?"))

for i in range(nb_traces_to_generate):

    m = randint(2**(nb_bit-1), (2**nb_bit)-1)
    n = randint(2**(nb_bit-1), (2**nb_bit)-1)
    d = randint(2**(nb_bit-1), (2**nb_bit)-1)

    res, t_s, t_m = square_multiply_always(m,n,d,wordsize)
    with open(f'../traces/traces_square_mult_{i}.pkl', 'wb') as f:
        pickle.dump((t_s, t_m), f)
    with open(f'../variables/variables_{i}.pkl', 'wb') as f:
        pickle.dump((m, n, d), f)
    
    print(f"Saved traces_square_mult_{i}.pkl successfully.")
    print(f"Saved variables_{i}.pkl successfully.")