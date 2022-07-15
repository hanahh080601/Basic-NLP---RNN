from rnn.config.config import *

class Dataset:
    '''
    Load data from data path, preprocess (tokenize & one-hot encode) and get data in array type.
    '''
    def __init__(self, data_train_url = Config.data_train_url, data_val_url = Config.data_val_url):
        with io.open (data_train_url, 'r') as f:
            self.text_train = f.read()
        with io.open (data_val_url, 'r') as f:
            self.text_val = f.read()
        self.chars = tuple(set(self.text_train))
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

    def char_tokenize(self):
        self.train_data = np.array([self.char2int[ch] for ch in self.text_train])
        self.val_data = np.array([self.char2int[ch] for ch in self.text_val])

    def one_hot_encode(self, arr, n_labels):
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
        one_hot = one_hot.reshape((*arr.shape, n_labels))
        return one_hot

    def get_data(self):
        self.char_tokenize()
        return self.train_data, self.val_data