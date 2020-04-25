Extending Linguistic Style Transfer for "Trumpifying" Text
=============

How it Works
------------
This repository is largely forked from https://github.com/agaralabs/transformer-drg-style-transfer. However, because we completed this project on Colab, some features have been adapted. Notably, bash scripts exist to run Python files on Colab. Specifically, train.sh trains the BERT Classifier in Part (1), dg.sh trains the OpenAI GPT Delete and Generate model in Part (4) and drg.sh trains the OpenAI GPT Delete, Retrieve, Generate model in Part (4). The pickle file senlist.data contains the results the processed sentences in Part (2) to reduce time spent retraining. Head-selection-2.ipynb should be used instead of Head-selection.ipynb.

In terms of directory structure, our LSTM / RNN baseline model (sourced from https://github.com/rpryzant/delete_retrieve_generate) is in baseline/. Our raw data, preprocessed data, and processed and tokenized data can be found under data/. The trained Delete and Generate model weights and parameters can be found under dg_model_weights/ and the trained Delete Retrieve Generate model weights and parameters can be found under dgr_model_weights/. Losses for both aforementioned models as well as the BERT Classifier are in losses/. The trained BERT Classifier can be found under models/.

Finally, some notebooks may not run on Jupyter because of !sudo apt-get install commands, which to our knowledge, is only a feature on Colab. There may also be some Path names that have to be changed before running.

Again, this repository is sourced from https://github.com/agaralabs/transformer-drg-style-transfer. The instructions provided at the source repository work for the most part on this repository, with modifications listed above.

Technical Requirements
----------------------

**Instructions and platform specifications**

>**Languages and tools used:**
>
>- torch    >= 1.0.1.post2
>- pandas   >= 0.23.4
>- numpy    >= 1.15.4
>- python   >= 3.7.1
>- tqdm     >= 4.28.1 
>- boto3    >= 1.9.107
>- requests >= 2.21.0
>- regex    >= 2019.2.21
>- git-lfs
>
>**Instructions**
>
>1. Clone this repository.
>2. Follow the instructions listed at https://github.com/agaralabs/transformer-drg-style-transfer, noting the exceptions lsited above.
