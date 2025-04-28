import pickle
import numpy as np

def load_traces_and_variables():
    ########## LOADING TRACES AND VARIABLES ###########
    t_s = [[] for _ in range(10)] 
    t_m = [[] for _ in range(10)] 
    m = [[] for _ in range(10)] 
    n = [[] for _ in range(10)] 
    d = [[] for _ in range(10)] 

    for i in range(10):
        with open(f'traces/traces_square_mult_{i}.pkl', 'rb') as f:
            t_s[i], t_m[i] = pickle.load(f)
        
        with open(f'variables/variables_{i}.pkl', 'rb') as f:
            m[i], n[i], d[i] = pickle.load(f)

    ##########        CHANGING FORMAT       ###########
    ######## (10,2047,64,64) -> (10,2047,4096) ########
    #######         LISTS -> NUMPY ARRAY        #######

    t_s_4096 = [[[] for _ in range(2048)] for _ in range(10)]
    t_m_4096 = [[[] for _ in range(2048)] for _ in range(10)]

    for i in range(len(t_s)):
        for j in range(len(t_s[i])): 
            for k in range(len(t_s[i][j])):
                for l in range(len(t_s[i][j][k])):
                    t_s_4096[i][j].append(t_s[i][j][k][l]/(2*len(t_s[1][1])))
                    t_m_4096[i][j].append(t_m[i][j][k][l]/(2*len(t_s[1][1])))

    t_s_4096 = [trace_set[1:] for trace_set in t_s_4096]  # Removes first trace from each set (only 1 element)
    t_m_4096 = [trace_set[1:] for trace_set in t_m_4096]

    t_s_4096_np = np.array(t_s_4096, dtype=np.float32)  
    t_m_4096_np = np.array(t_m_4096, dtype=np.float32)

    return t_s_4096_np, t_m_4096_np, m, n, d

def write_preprocessed_data(t_s_4096_np, t_m_4096_np):
    ########### WRITING DATA IN ATTACK AND TRAINING SETS WITH RIGHT FORMAT ##########
    for i in range(len(t_s_4096_np)-1):
        with open(f'training_set/training_squares{i}.pkl', 'wb') as f:
            pickle.dump((t_s_4096_np[i]), f)
        with open(f'training_set/training_mult{i}.pkl', 'wb') as f:
            pickle.dump((t_m_4096_np[i]), f) 
    with open(f'attack_set/attack_square{9}.pkl', 'wb') as f:
            pickle.dump((t_s_4096_np[9]), f)

    with open(f'attack_set/attack_mult{9}.pkl', 'wb') as f:
            pickle.dump((t_m_4096_np[9]), f)

def load_preprocessed_data():
    ########### LOADING DATA IN RIGHT FORMAT ##########
    t_s_4096_np = [[[] for _ in range(2048)] for _ in range(10)]
    t_m_4096_np = [[[] for _ in range(2048)] for _ in range(10)]

    m = [[] for _ in range(10)]
    n = [[] for _ in range(10)]
    d = [[] for _ in range(10)]

    for i in range(10):
        if i == 9:  # Assuming the last one is the attack set
            with open(f'attack_set/attack_mult{i}.pkl', 'rb') as f:
                t_m_4096_np[i] = np.array(pickle.load(f))
            with open(f'attack_set/attack_square{i}.pkl', 'rb') as f:
                t_s_4096_np[i] = np.array(pickle.load(f))
        else:
            with open(f'training_set/training_mult{i}.pkl', 'rb') as f:
                t_m_4096_np[i] = np.array(pickle.load(f))
            with open(f'training_set/training_squares{i}.pkl', 'rb') as f:
                t_s_4096_np[i] = np.array(pickle.load(f))

        with open(f'variables/variables_{i}.pkl', 'rb') as f:
            m[i], n[i], d[i] = pickle.load(f)

    return t_s_4096_np, t_m_4096_np, m, n, d