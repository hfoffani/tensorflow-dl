

# Deep Learning Examples in Tensorflow.


This repo contains a handful of Deep Learning examples written in Python Jupyter notebooks.


### Directory

 - `1_notmnist.ipynb`
    Processes MNIST data.
 - `2_fullyconnected.ipynb`
    Uses a fully connected layer for prediction.
 - `3_regularization.ipynb`
    Adds regularization.
 - `4_convolutions.ipynb`
    Implements Convoluational Network.
 - `5_word2vec.ipynb`
    Implement Word2Vec.
 - `6_lstm.ipynb`
    Basic LSTM network.
 - `62a_lstm.ipynb`
    An LSTM network.
 - `62b_lstm.ipynb`
    An enhanced LSTM.
 - `63_lstm.ipynb`
    Another LSTM.
 - `6_translate.ipynb`
    A NN translator.

There are other notebooks as well.


### Installation.

For Mac follow the following instructions.

Update XCode and install `brew` and `pyenv`

    xcode-select --install
    brew

    brew install pyenv
    brew install pyenv-virtualenv


Due to Matplotlib you will need to install a "Framework" Python.

    git clone https://github.com/s1341/pyenv-alias.git ~/.pyenv/plugins/pyenv-alias

    VERSION_ALIAS="framework_352" PYTHON_CONFIGURE_OPTS="--enable-framework CC=clang CFLAGS=-I$(brew --prefix openssl)/include LDFLAGS=-L$(brew --prefix openssl)/lib" pyenv install 3.5.2

    pyenv virtualenv framework_352 tf


Modify your `.bash_profile` for Jupyter. In `~/.bash_profile` add the following lines:

    # needed for Jupyter
    export LC_ALL=en_US.UTF-8

Now you are able to Install all the Python packages.

    pip install -r requirements.txt
    pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py3-none-any.whl


