from rnn.config.config import *

class DataLoader:
    '''
    Load data from dataset in batches (batches = n_seqs * n_steps)
    '''
    def __init__(self, train, val):
        self.train = train
        self.val = val

    def __call__(self, arr, n_seqs, n_steps):
        '''
        Create a generator that returns batches of size
        n_seqs x n_steps from arr.
        
        Arguments
        ---------
        arr: np.array
            Array you want to make batches from
        n_seqs: int
            Batch size, the number of sequences per batch
        n_steps: int
            Number of sequence steps per batch
        '''
        batch_size = n_seqs * n_steps
        n_batches = len(arr) // batch_size
        arr = arr[:n_batches * batch_size]
        arr = arr.reshape((n_seqs, -1))
        
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n: n + n_steps]
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield x, y