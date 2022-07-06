import copy
import random
import numpy as np

from utils import read_flowchart_jsons, read_dialog_jsons, build_vocab, create_batches, vectorize_text

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SiameseBertBatch(object):

    def __init__(self, data, flowcharts, glob, start, end, test=False):
        
        if test:
            # in test batch size is set to be 1
            # all negative examples to be added for evaluation
            flowchart_name = data.flowchart_names[start]
            no_of_nodes_in_flowchart = len(flowcharts.get_all_node_ids(flowchart_name))
            repeats = no_of_nodes_in_flowchart
        else:
            repeats = 2
        
        user_utts = []
        agent_utts = []

        user_utts = np.repeat(data.user_utt_as_text[start:end], repeats=repeats, axis=0)
        agent_utts = np.repeat(data.user_utt_as_text[start:end], repeats=repeats, axis=0)

        self.flowchart_names = np.repeat(data.flowchart_names[start:end], repeats=repeats, axis=0)
        self.response_node_ids = np.repeat(data.response_node_ids[start:end], repeats=repeats, axis=0)
        
        # for each (user-utt,agent-utt) pair:
        #   pick the positive example and a random negative example based on the (flowchart, response_node_id) pair

        node_texts = []
        edge_texts = []

        self.gt_labels = []
        self.printable_text = []

        if test:

            response_node_id = data.response_node_ids[start]
            name = data.flowchart_names[start]
            
            flowchart_node_ids = flowcharts.get_all_node_ids(name)
            for flowchart_node_id in flowchart_node_ids:
                
                node_text =  flowcharts.get_parent_node_text(name, flowchart_node_id)
                node_texts.append(node_text)
                
                edge_text =  flowcharts.get_edge_from_parent(name, flowchart_node_id)
                edge_texts.append(edge_text)
                
                if flowchart_node_id == response_node_id:
                    self.gt_labels.append(1)
                else:
                    self.gt_labels.append(0)
            
                printable_text = {
                    'node_text': node_text,
                    'edge_text': edge_text,
                    'agent_utt': data.agent_utt_as_text[start],
                    'user_utt' : data.user_utt_as_text[start]
                }
                self.printable_text.append(copy.deepcopy(printable_text))
        else:

            for idx, response_node_id in enumerate(self.response_node_ids):
                name = self.flowchart_names[idx]

                if idx%2==0: # add true label
                    
                    node_text =  flowcharts.get_parent_node_text(name, response_node_id)
                    node_texts.append(node_text)
                    
                    edge_text =  flowcharts.get_edge_from_parent(name, response_node_id)
                    edge_texts.append(edge_text)

                    self.gt_labels.append(1)

                else: # add false label
                    
                    random_node_id = random.choice(list(flowcharts.get_all_node_ids(name)))
                    while random_node_id == response_node_id:
                        random_node_id = random.choice(list(flowcharts.get_all_node_ids(name)))

                    node_text =  flowcharts.get_parent_node_text(name, random_node_id)
                    node_texts.append(node_text)
                    
                    edge_text =  flowcharts.get_edge_from_parent(name, random_node_id)
                    edge_texts.append(edge_text)
                    

                    self.gt_labels.append(0)
        
        self.batch_sentids, self.batch_segids, self.batch_mask = self.tokenize_for_bert(user_utts, agent_utts, node_texts, edge_texts)

    def tokenize_for_bert(self, user_utts, agent_utts, node_texts, edge_texts):

        #for user_utt, agent_utt, node_text, edge_text in zip(user_utts, agent_utts, node_texts, edge_texts):
        dialog_tokens = [bert_tokenizer.tokenize(user_utt + " " + agent_utt) for user_utt, agent_utt in zip(user_utts, agent_utts)]
        dialog_tokenids = [bert_tokenizer.convert_tokens_to_ids(x) for x in dialog_tokens]

        node_tokens = [bert_tokenizer.tokenize(node_text + " " + edge_text) for node_text, edge_text in zip(node_texts, edge_texts)]
        node_tokensids = [bert_tokenizer.convert_tokens_to_ids(x) for x in node_tokens]

        joint_sentids = [bert_tokenizer.build_inputs_with_special_tokens(x,y) for x,y in zip(dialog_tokenids, node_tokensids)]
        joint_segids = [bert_tokenizer.create_token_type_ids_from_sequences(x,y) for x,y in zip(dialog_tokenids, node_tokensids)]

        lengths = [len(x) for x in joint_sentids]
        maxlen = max(lengths)
        
        batch_sentids = [x + [0]*(maxlen-ll) for x,ll in zip(joint_sentids, lengths)]

        batch_segids = [x + [0]*(maxlen-ll) for x,ll in zip(joint_segids, lengths)]
        
        batch_mask = [[int(x > 0) for x in sent_id] for sent_id in batch_sentids]
        
        return batch_sentids, batch_segids, batch_mask