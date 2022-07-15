from rnn.config.config import *

class RNN(nn.Module):
    def __init__(self, vocab_size, n_steps=Config.n_steps, n_hidden=Config.n_hidden, n_layers=Config.n_layers,
                    drop_prob=Config.dropout, lr=Config.lr):
        super().__init__()
        self.vocab_size = vocab_size
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr      
        self.lstm = nn.LSTM(vocab_size, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)        
        self.dropout = nn.Dropout(drop_prob)      
        self.fc = nn.Linear(n_hidden, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        ''' 
        Initialize weights for fully connected layer 
        '''
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)
        
    def init_hidden(self, n_seqs):
        ''' 
        Initializes hidden state 
        '''
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())

    def forward(self, x, hc):
        ''' 
        Forward pass through the network. 
        These inputs are x, and the hidden/cell state `hc`. 
        '''
        x, (h, c) = self.lstm(x, hc)
        x = self.dropout(x)
        x = x.reshape(x.size()[0] * x.size()[1], self.n_hidden)
        x = self.fc(x)
        return x, (h, c)