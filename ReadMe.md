#Neural Reduplication

##About
This repo contains code for learning reduplication (or more generally phonological transformations) with encoder-decoder networks. Results from experiments run with this code are reported in Nelson, Dolatian, Rawksi, and Prickett (2020)

##Contents
`src` contains all necessary code

`data` contains sample reduplication data files, created by transducers generated from a typology of natural language reduplication patterns (Dolatian and Heinz, 2019). 

You can create your own data files by following the format shown in `data`. Each line represents a single mapping and is structured "input(\tab)-->(\tab)output"

`hyperparameters` contains a sample hyperparameter file, `example.txt`. You can create your own hyperparameter file by following the format in the sample file. 

*   `input_file` - path to the file containing the training data
*   `teacher_force` - boolean, whether or not to use teacher forcing, default `True`
*   `attention` - boolean, whether or not to use a global attention mechanism, default `True`
*   `attention_type` - string indicating what type of attention to use, options are `'dot'` or `'weighted'`, ignored if `attention=False`, default `'dot'`
*   `recurrent_type` - string indicating type of recurrent network to use for encoder and decoder, options are `'RNN'`, `'GRU'`, and `'LSTM'`, default `'GRU'`
*   `embedding_size` - integer size of phoneme representations, default `16`
*   `hidden_size` - integer size of encoder and decoder hidden states, default `64`
*   `num_epochs` - integer number of epochs to train the model, default `50`
*   `print_freq` - integer frequency with which to print loss during training, default `5`
*   `batch_size` - integer batch_size used during training, default `100`
*   `learning_rate` - float initial learning rate (used with Adam optimization), default `0.001`
*   `dropout_prob` - float dropout probability during training, default `0.1` 


##Running the models
Requirements: Python 3.6+ with NumPy and Pytorch (1.0 or later)

A sample version of the model can be run from the command line with:

`python src/main.py hyperparameters/example.txt`




