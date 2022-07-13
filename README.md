# SA-FloNet

This repository contains the implementation of SA-FloNet.

# Reproducing Test Results:

1. Clone this repo
2. Set up a Python environment using requirements.txt (we use Python 3.6 for all our experiments)
3. Download pre-trained glove embeddings and unzip them to code/glove6B/
4. The zip file contains the models files (with retriever and generator checkpoints) for both settings. Download and unzip it to data/model/
5. Run the inference script with the following commands. The final line of the output file will include the BLEU, R@1 and PPL metrics.

## For S-Flo

```
python sa-flonet.py --dropout=0.0 --lr=0.0001 --gpt-lr=0.25e-5 \
--save-name=SAFlonetSFloTestEval \
--num-epochs=0 --dialog-dir='../data/dialogs/' --domain='in_domain_hard' \
--cached-dialog-path='../data/saved_data/cached_in_domain_hard_dialogs.pkl' --saved-glove-path=./glove6B/ \
--si_model_checkpoint="../data/model/SAFlonetSFloBestModel/Scorer_best_checkpoint.pth.tar" \
--gpt_model_checkpoint="../data/model/SAFlonetSFloBestModel/GPT_best_checkpoint/" \
--inference=1 --model=gpt2 --max_history=900 --num-epochs=0 --history_length=600 --max_length=60 --personality_length=200 \
>outputs/SAFlonetSFloTestEval.out 2>outputs/SAFlonetSFloTestEval.err
```

## For U-Flo

```
python sa-flonet.py --dropout=0.0 --lr=0.0001 --gpt-lr=0.25e-5 \
--save-name=SAFlonetUFloTestEval \
--num-epochs=0 --dialog-dir='../data/dialogs/' --cached-dialog-path='../data/saved_data/cached_out_domain_dialogs.pkl' --domain='out_domain' \
--saved-glove-path=./glove6B/ \
--si_model_checkpoint="../data/model/SAFlonetUFloBestModel/Scorer_best_checkpoint.pth.tar" \
--gpt_model_checkpoint="../data/model/SAFlonetUFloBestModel/GPT_best_checkpoint/" \
--inference=1 --max_length=60 --model=gpt2 --max_history=900 --num-epochs=0 --history_length=600 --emb-size=200 --hidden-size=600 --personality_length=200 \
>outputs/SAFlonetUFloTestEval.out 2>outputs/SAFlonetUFloTestEval.err
```
6. The metrics are printed at the end of the output files. We use average top perplexity (second to last PPL score) for our perplexity results.

# SA-FloNet outputs

The retriever and generator output along with ground truth annotations for each context response pair in the FloDial test split are provided in sa-flonet_test_output/ for both settings.



