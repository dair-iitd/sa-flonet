# SA-FloNet

This repository contains the implementation of SA-FloNet.

# Reproducing Test Results:

1. Clone this repo
2. Set up a Python environment using requirements.txt (we use Python 3.6 for all our experiments)
3. Download [pre-trained glove embeddings](https://nlp.stanford.edu/data/glove.6B.zip) and unzip them to ```code/glove6B/```
4. The [link](https://drive.google.com/drive/folders/1-jzJJXG34RR581-0yY5u9ZDKCm6U-z0l?usp=sharing) contains the best SAFlonet model files (with retriever and generator checkpoints) used in paper results in both settings. Download and move the contents to ```data/model/```
5. Run the inference script with the following commands (evaluation time for both settings = 2 hours). For the below commands, it is assumed that the inference scripts are in a subfolder of ```code``` e.g. ```code/hpc/```. The output files will be generated in ```code/output/```. The final line of the output file will include the BLEU, R@1 and PPL metrics

## For S-Flo

```
python sa-flonet.py --dropout=0.0 --lr=0.0001 --gpt-lr=0.25e-5 \
--save-name=SAFlonetSFloBestModelTestEval \
--num-epochs=0 --dialog-dir='../data/dialogs/' --domain='in_domain_hard' \
--cached-dialog-path='../data/saved_data/cached_in_domain_hard_dialogs.pkl' --saved-glove-path=./glove6B/ \
--si_model_checkpoint="../data/model/SAFlonetSFloBestModel/Scorer_best_checkpoint.pth.tar" \
--gpt_model_checkpoint="../data/model/SAFlonetSFloBestModel/GPT_best_checkpoint/" \
--inference=1 --model=gpt2 --max_history=900 --history_length=600 --max_length=60 --personality_length=200 \
>outputs/SAFlonetSFloBestModelTestEval.out 2>outputs/SAFlonetSFloBestModelTestEval.err
```

## For U-Flo

```
python sa-flonet.py --dropout=0.0 --lr=0.0001 --gpt-lr=0.25e-5 \
--save-name=SAFlonetUFloBestModelTestEval \
--num-epochs=0 --dialog-dir='../data/dialogs/' --cached-dialog-path='../data/saved_data/cached_out_domain_dialogs.pkl' --domain='out_domain' \
--saved-glove-path=./glove6B/ \
--si_model_checkpoint="../data/model/SAFlonetUFloBestModel/Scorer_best_checkpoint.pth.tar" \
--gpt_model_checkpoint="../data/model/SAFlonetUFloBestModel/GPT_best_checkpoint/" \
--inference=1 --max_length=60 --model=gpt2 --max_history=900 --history_length=600 --emb-size=200 --hidden-size=600 --personality_length=200 \
>outputs/SAFlonetUFloBestModelTestEval.out 2>outputs/SAFlonetUFloBestModelTestEval.err
```

# SA-FloNet outputs

The retriever and generator output along with ground truth annotations for each context response pair in the FloDial test split are provided in sa-flonet_test_output/ for both settings.

# Training SA-FloNet from scratch

1. Clone this repo
2. Set up a Python environment using requirements.txt (we use Python 3.6 for all our experiments)
3. Download [pre-trained glove embeddings](https://nlp.stanford.edu/data/glove.6B.zip) and unzip them to ```code/glove6B/```
4. The [link]([link](https://drive.google.com/drive/folders/1-jzJJXG34RR581-0yY5u9ZDKCm6U-z0l?usp=sharing)) contains the pre-trained model files (retriever and generator) for both settings that are used for training SA-FloNet. Download and move them to ```data/model/```
5. Run the training script with the following commands (training time for both settings = 32 hours). For the below commands, it is assumed that the inference scripts are in a subfolder of ```code``` e.g. ```code/hpc/```. The output files will be generated in ```code/output/```.

## For S-Flo

```
python sa-flonet.py --dropout=0.0 --lr=0.0001 --gpt-lr=0.25e-5 \
--save-name=SAFlonetSFloTrain \
--num-epochs=100 --dialog-dir='../data/dialogs/' --domain='in_domain_hard' \
--cached-dialog-path='../data/saved_data/cached_in_domain_hard_dialogs.pkl' --saved-glove-path=./glove6B/ \
--si_model_checkpoint="../data/model/SFloPretrainedModel/Scorer_best_checkpoint.pth.tar" \
--gpt_model_checkpoint="../data/model/SFloPretrainedModel/GPT_best_checkpoint/" \
--inference=0 --model=gpt2 --max_history=900  --max_length=60 --history_length=600 --personality_length=200 \
>outputs/SAFlonetSFloTrain.out 2>outputs/SAFlonetSFloTrain.err
```

## For U-Flo

```
python sa-flonet.py --dropout=0.0 --lr=0.0001 --gpt-lr=0.25e-5 \
--save-name=SAFlonetUFloTrain \
--num-epochs=100 --dialog-dir='../data/dialogs/' --domain='out_domain' \
--cached-dialog-path='../data/saved_data/cached_out_domain_dialogs.pkl' --saved-glove-path=./glove6B/ \
--si_model_checkpoint="../data/model/UFloPretrainedModel/Scorer_best_checkpoint.pth.tar" \
--gpt_model_checkpoint="../data/model/UFloPretrainedModel/GPT_best_checkpoint/" \
--inference=0 --model=gpt2 --max_history=900 --max_length=60 --emb-size=200 --hidden-size=600 --history_length=600 --personality_length=200 \
>outputs/SAFlonetUFloTrain.out 2>outputs/SAFlonetUFloTrain.err
```
6. After training is complete, to get the test results, run the inference script (evaluation time for both settings = 2 hours) with the following commands. For the below commands, it is assumed that the inference scripts are in a subfolder of ```code``` e.g. ```code/hpc/```. The output files will be generated in ```code/output/```. The metrics are printed at the end of the output files.

## For S-Flo

```
python sa-flonet.py --dropout=0.0 --lr=0.0001 --gpt-lr=0.25e-5 \
--save-name=SAFlonetSFloTestEval \
--num-epochs=0 --dialog-dir='../data/dialogs/' --domain='in_domain_hard' \
--cached-dialog-path='../data/saved_data/cached_in_domain_hard_dialogs.pkl' --saved-glove-path=./glove6B/ \
--si_model_checkpoint="../data/model/SAFlonetSFloTrain/Scorer_best_checkpoint.pth.tar" \
--gpt_model_checkpoint="../data/model/SAFlonetSFloTrain/GPT_best_checkpoint/" \
--inference=1 --model=gpt2 --max_history=900 --history_length=600 --max_length=60 --personality_length=200 \
>outputs/SAFlonetSFloTestEval.out 2>outputs/SAFlonetSFloTestEval.err
```

## For U-Flo

```
python sa-flonet.py --dropout=0.0 --lr=0.0001 --gpt-lr=0.25e-5 \
--save-name=SAFlonetUFloTestEval \
--num-epochs=0 --dialog-dir='../data/dialogs/' --domain='out_domain' \
--cached-dialog-path='../data/saved_data/cached_out_domain_dialogs.pkl' --saved-glove-path=./glove6B/ \
--si_model_checkpoint="../data/model/SAFlonetUFloTrain/Scorer_best_checkpoint.pth.tar" \
--gpt_model_checkpoint="../data/model/SAFlonetUFloTrain/GPT_best_checkpoint/" \
--inference=1 --max_length=60 --model=gpt2 --max_history=900 --history_length=600 --emb-size=200 --hidden-size=600 --personality_length=200 \
>outputs/SAFlonetUFloTestEval.out 2>outputs/SAFlonetUFloTestEval.err
```

For both inference and training, if doing multiple runs make sure to rename the save-name, and output file names to prevent overwriting of earlier run models and outputs. For inference runs, check si_model_checkpoint and gpt_model_checkpoint to map to the checkpoints from the correct training runs.

# Model performance on individual context-response pairs

After each inference run, directory ```logs/``` will be created at the top level alongside ```code/``` and ```data/```. Inside it will be a directory with a name beginning with the save-name string of the run. This will contain the json output file ```test-output-1.json``` contains details of how the model performed for every test context-response pair. This may be used for further analysis. This is similar to the files in ```sa-flonet_test_output/``` with more details. The same keys represent the same details in both the output files generated in ```logs/``` and ```sa-flonet_test_output/```. A similar directory is also created in ```logs/``` for every training run but this can be ignored for test analysis.
