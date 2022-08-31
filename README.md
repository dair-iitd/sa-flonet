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

# SemBERT for NLI

Due to computational limitations, we precompute the scores given by a fine-tuned SemBERT model. For every flowchart, these scores are computed for all possible pairs of questions and answers to account for all possibilities.

Our fork of SemBERT for the purposes of SA-FloNet can be found [here](https://github.com/dair-iitd/SemBERT).

The new_dialog_data/ folder contains the data used for fine-tuning using in the paper.

The scripts for fine-tuning and generating scores are in the hpc/ folder. 

These scripts output the fine-tuned model in new_models/ in the appropriate subfolder. Moreover, the scripts are also output the scores generated by the fine-tuned model in the same folder as the fine-tuned model. 

These cached scores are used during training and evaluation in SA-FloNet. Copying the scores generated in the fine-tuned model's folder into the appropriate location in code/cached-nli-scores/ and using the convert_scores_to_cached_dicts.py script transforms the scores into a dict (say nli_scores) so that given a question "q" and an answer "a" the scores for contradiction, entailment and neutral can be found as nli_scores["q"]["a"][0], nli_scores["q"]["a"][1] and nli_scores["q"]["a"][2] respectively.

It simulates the running of the fine-tuned SemBERT model during SA-FloNet training/evaluation without having to load a SemBERT model into the memory.

# Fine-tuning SemBERT with your own data for SA-FloNet

1. Clone the SemBERT [fork](https://github.com/dair-iitd/SemBERT) for SA-FloNet.
2. Setup a python environment using the instructions in the readme. Download and setup the trained SNLI and SRL model as mentioned in the **Evaluation using raw data (with real-time semantic role labeling)** section.
3. Divide your data into train and dev splits. It should have the same columns in the same order as shown [here](https://github.com/dair-iitd/SemBERT/blob/master/new_dialog_data/indomain/circa_and_full_DS_data/dev.tsv).
4. Run the training script with the following commands. Modify the ```data_dir``` and ```output_dir``` as appropriate. The other parameters can be varied according to requirements or computational capacity. A file containing the validation performance will also be generated in ```output_dir```.

```
python run_classifier_snli_online_tagging.py \
--data_dir new_dialog_data/outdomain/circa_and_full_DS_data \
--task_name snli \
--train_batch_size 6 \
--max_seq_length 128 \
--bert_model snli_sembert_model/ \
--learning_rate 2e-5 \
--num_train_epochs 2 \
--do_train \
--do_eval \
--do_predict \
--do_lower_case \
--max_num_aspect 3 \
--output_dir new_models/outdomain/circa_and_full_DS_data_FT \
--tagger_path srl_model/
```

5. Run the prediction script with the following commands. Set ```data_dir``` to the new_dialog_data/indomain/Flonet_indomain_data/ or the new_dialog_data/outdomain/Flonet_outdomain_data/ folder for S-Flo and U-Flo domain respectively. Set the ```output_path``` to the same as that in the train script. Set ```eval_model``` to the best performing model file in ```output_path```. The names of the model files will be (epoch number)_pytorch_model.bin according to the epoch. Set ```domain``` to S-Flo or U-Flo. Set ```dataset``` to train, val or test to cache the scores for that dataset. Cached scores for all the three splits are needed for running SA-Flonet. The cached score files will be generated in ```output_dir``` with names according to their domain and split ending in score_only.json

```
python run_snli_predict_scores.py \
--data_dir new_dialog_data/outdomain/Flonet_outdomain_data \
--task_name snli \
--eval_batch_size 32 \
--max_seq_length 128 \
--max_num_aspect 3 \
--do_lower_case \
--bert_model snli_sembert_model/ \
--output_dir new_models/outdomain/circa_and_full_DS_data_FT \
--tagger_path srl_model \
--do_predict \
--eval_model 1_pytorch_model.bin \
--dataset test \
--domain U-Flo
```

6. Copy the score_only.json files into the directory for this repo in an appropriate subfolder in code/cached-nli-scores/. Make sure to give the subfolder in both domains the same name. Modify the convert_scores_to_cached_dicts.py script by changing the paths as appropriate and run it every domain and split. The necessary dicts will be generated and saved in json files.

7. In the SA-FloNet training and evaluation scripts, modify the ```nli-folder``` parameter to the name of the subfolder for your newly generated cached scores in the previous step. SA-FloNet will use the scores of the new fine-tuned model. Run SA-FloNet with other parameters as described above.
