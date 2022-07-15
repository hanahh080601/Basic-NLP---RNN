# Basic-NLP-RNN
# Build a basic Recurrent Neural Network using Pytorch.

## Tasks 
* Preprocessing data
  * Loading data.
  * Create character-level vocabulary from training text.
  * Transform data.
* Building model
  * Construct model with LSTM, Dropout & Linear layer.
  * Init hidden state.
  * Override forward function.
* Training
  * Define criterion, optimizer.
  * One-hot encode data
  * Execute (train, validate).
  * Early stopping. 
  * Save model.
* Evaluating
  * Generate n next characters with given prime "Juliet".
  * Result with 1000 next characters with given prime "Juliet":  
    ```bash 
    Juliet of my like,
    The words with a traches; but all my house
    With some mighty and such a man,
    If stands in tentience to the countery
    To this me sense of sected sight.

    MARINA:
    Why, she hath better stand all as the present serve.

    Second Gentleman:
    I have this sorrow strongs thou, shall to stand them:
    The show with my belessiones, as thou art sent
    We will be send. He woulds here is there the most the
    stope, and thanks that humble manys by sufficed with
    And tell my husband she is not the stain,
    As it before thee so thou and the some of here,
    And better that were something to the service.

    CRESSIDA:
    What hath me be this time? the man to mildent
    How much shall help here in mine the chill were,
    The wantom to her truth, word to secure the course
    Of the caure is made a man that spiril there and make
    The change they will the matter of my stang.
    ...
    SUFFOLK:
    We will not been short on thy both a morn
    That I am steal. What is myself, make a stain to thy hand,
    With honest said and sease their brother world
    ```


## Installation

Clone the repo from Github and pull the project.
```bash
git clone https://github.com/hanahh080601/Basic-NLP-RNN.git
git checkout hanlhn/rnn
git pull
cd rnn/rnn
poetry install
poetry config virtualenvs.in-project true
poetry update
```

# Project tree 
.  
├── rnn          
│     ├── .venv             
│     ├── poetry.lock    
│     ├── pyproject.toml   
│     ├── README.rst  
│     └── rnn   
│           ├── __pycache__  
│           ├── data            
│           │     ├── dataset.py  
│           │     ├── dataloader.py  
│           │     ├── shakespeare_train.txt    
│           │     └── shakespeare_valid.txt                
│           ├── models        
│           │      ├── RNN.py    
│           │      └──rnn.net    
│           ├── train          
│           │      └── train.py   
│           ├── generate              
│           │      └── generate.py    
│           ├── notebooks       
│           │      ├── rnn_1_epoch.py      
│           │      └── rnn_25_epoch.py    
│           ├── tests       
│           │      ├── __init__.py        
│           │      └── test_rnn.py     
│           ├── __init__.py       
│           └── config              
│                  └── config.py     
├── .gitignore                    
└── README.md   

## Usage: 
```bash
cd rnn/rnn/generate
python generate.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Author
[Lê Hoàng Ngọc Hân - Đại học Bách Khoa - Đại học Đà Nẵng (DUT)](https://github.com/hanahh080601) 
