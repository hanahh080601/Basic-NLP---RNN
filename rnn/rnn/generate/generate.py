from rnn.config.config import *
from rnn.data.dataset import Dataset
from rnn.models.RNN import RNN

def predict(net, char, h=None, cuda=False, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        if cuda:
            net.cuda()
        else:
            net.cpu()
        
        if h is None:
            h = net.init_hidden(1)
        
        x = np.array([[data.char2int[char]]])
        x = data.one_hot_encode(x, len(data.chars))
        inputs = torch.from_numpy(x)
        if cuda:
            inputs = inputs.cuda()
        
        h = tuple([each.data for each in h])
        out, h = net.forward(inputs, h)

        p = F.softmax(out, dim=1).data
        if cuda:
            p = p.cpu()
        
        if top_k is None:
            top_ch = np.arange(len(data.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
            
        return data.int2char[char], h

def sample(net, size, prime='The', top_k=None, cuda=False):
    '''
    Generate the next `size` characters from given `prime`
    '''
    if cuda:
        net.cuda()
    else:
        net.cpu()

    net.eval()
    
    # Run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)
    
    # Pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

# Here we have loaded in a model that trained over 1 epoch `rnn_1_epoch.net`
with open(Config.model_path, 'rb') as f:
    state_dict = torch.load(f)
    
data = Dataset()
loaded = RNN(vocab_size=len(data.chars))
loaded.load_state_dict(state_dict)

# Change cuda to True if you are using GPU!
print(sample(loaded, 1000, cuda=False, top_k=5, prime="Juliet"))