from requirements import *
from parameters import *
from writing_loading_traces import load_traces_and_variables, write_preprocessed_data, load_preprocessed_data

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=100, trace_length=4096, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        self.trace_length = trace_length
        self.batch_size = batch_size
        self.list_IDs = list_IDs  # [[4096 points]*2047]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        num_exponentiations = self.list_IDs[0].shape[0]  # 7 for 'train'
        traces_per_exp = self.list_IDs[0].shape[1]       # 2047
        total_traces = num_exponentiations * traces_per_exp
        return int(np.floor(total_traces / self.batch_size))

    def __getitem__(self, index):
        """Generate pairs of traces."""
        X1_batch = np.zeros((self.batch_size, self.trace_length, 1))
        X2_batch = np.zeros((self.batch_size, self.trace_length, 1))
        
        for i in range(self.batch_size):
            # Randomly pick two traces (same exponentiation)
            exp_idx = np.random.randint(0,self.list_IDs[0].shape[0])
            idx_mult = np.random.randint(0, self.list_IDs[0][exp_idx].shape[0]-1)
            # idx_square = idx_mult + 1
            idx_square = np.random.randint(0, self.list_IDs[1][exp_idx].shape[0]) 
            
            X1_batch[i] = self.list_IDs[0][exp_idx][idx_mult].reshape(-1,1)
            X2_batch[i] = self.list_IDs[1][exp_idx][idx_square].reshape(-1,1)
        
        # Liste remplie de 1 pour les labels pcq osef
        y_batch = np.ones((self.batch_size, self.trace_length, 1))
        return (X1_batch, X2_batch), y_batch

    def on_epoch_end(self):
        """Shuffle data after each epoch if needed."""
        if self.shuffle:
            np.random.shuffle(self.list_IDs)

def data_generators(new_data = False):
    # If we just acquired new traces then we need to get the right format 
    if new_data:
        t_s, t_m, m, n, d = load_traces_and_variables()
        write_preprocessed_data(t_s, t_m)
        t_s_4096_np, t_m_4096_np, m, n, d = load_preprocessed_data()
    
    # If we just want to work on already-processed traces we just need to load these.
    else:
        t_s_4096_np, t_m_4096_np, m, n, d = load_preprocessed_data()


    t_s_4096_np = np.array(t_s_4096_np)
    t_m_4096_np = np.array(t_m_4096_np)


    partition = {
        'train': np.array([t_s_4096_np[:7], t_m_4096_np[:7]]),   # First 7 exponentiation sets (each with 2048 traces)
        'validation': np.array([t_s_4096_np[7:9],  t_m_4096_np[7:9]]),
        'attack': np.array([t_s_4096_np[9:], t_m_4096_np[9:]])   # Last exponentiation set
    }


    # Generator parameters
    params = {
        'batch_size': batch_size,  
        'trace_length': trace_length,
        'shuffle': shuffle
    }

    params_attack = {
        'batch_size': batch_size_attack,  
        'trace_length': trace_length,
        'shuffle': shuffle
    }

    # Create generators
    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validation'], **params)
    testing_generator = DataGenerator(partition['attack'], **params_attack)
    return training_generator, validation_generator, testing_generator, m, n, d