from rnn.config.config import *
from rnn.data.dataset import Dataset
from rnn.data.dataloader import DataLoader
from rnn.models.RNN import RNN

def train(net, train_data, val_data, epochs=Config.epochs, n_seqs=Config.n_seqs, 
          n_steps=Config.n_steps, lr=Config.lr, clip=Config.clip, cuda=Config.cuda):
    ''' 
        Training a network 
    
        Arguments
        ----------------
        net: RNN network
        train_data: text data to train the network
        val_data: text data to validate the network
        epochs: Number of epochs to train
        n_seqs: Number of mini-sequences per mini-batch, aka batch size
        n_steps: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        cuda: Train with CUDA on a GPU
    '''
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    the_last_loss = 100
    patience = 10
    trigger_times = 0
    isStopped = False
    if cuda:
        net.cuda()
    
    counter = 0
    for e in range(epochs):
        h = net.init_hidden(n_seqs)
        if isStopped:
            break
        for x, y in data_loader(train_data, n_seqs, n_steps):
            counter += 1
            
            # One-hot encode our data and make them Torch tensors
            x = data.one_hot_encode(x, net.vocab_size)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            h = tuple([each.data for each in h])

            net.zero_grad()
            
            output, h = net.forward(inputs, h)
            loss = criterion(output, targets.view(n_seqs*n_steps))

            loss.backward()
            
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            opt.step()
            
            if counter % 10 == 0:
                
                val_h = net.init_hidden(n_seqs)
                val_losses = []
                for x, y in data_loader(val_data, n_seqs, n_steps):
                    x = data.one_hot_encode(x, net.vocab_size)
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                    
                    if cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    val_h = tuple([each.data for each in val_h])

                    output, val_h = net.forward(inputs, val_h)
                    val_loss = criterion(output, targets.view(n_seqs*n_steps))
                
                    val_losses.append(val_loss.item())
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

                the_current_loss = np.mean(val_losses)
                if the_current_loss > the_last_loss:
                    trigger_times += 1
                    print('trigger times: ', trigger_times)
                    if trigger_times >= patience:
                        print('Early stopping! at epoch {0}'.format(e))
                        isStopped = True
                        break

                else:
                    print('trigger times: 0')
                    trigger_times = 0
                    the_last_loss = the_current_loss
                    if not isStopped:
                        with open('models/rnn.net', 'wb') as f:
                            torch.save(net.state_dict(), f)
                        print('Validation loss {:.6f}.  Saving model ...'.format(the_current_loss))

# Define dataset & dataloader
data = Dataset()
train_data, val_data = data.get_data()
data_loader = DataLoader(train_data, val_data)

# Define and print the net
net = RNN(vocab_size=len(data.chars))
print(net)

# Training
train(net=net, train_data=train_data, val_data=val_data)