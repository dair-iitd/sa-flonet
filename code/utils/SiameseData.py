import copy
import random
import numpy as np

from utils import read_flowchart_jsons, read_dialog_jsons, build_vocab, create_batches, vectorize_text

class SiameseData(object):
    
    def __init__(self, dataJson, glob):

        self._user_utt = []
        self._user_utt_as_text = []
        self._user_utt_lengths = []

        self._agent_utt = []
        self._agent_utt_as_text = []
        self._agent_utt_lengths = []

        self._response_node_ids = []
        self._flowchart_names = []

        self._populate_data_from_json(dataJson, glob)
    
    @property
    def agent_utt(self):
        return self._agent_utt
    
    @property
    def agent_utt_as_text(self):
        return self._agent_utt_as_text
    
    @property
    def agent_utt_lengths(self):
        return self._agent_utt_lengths

    @property
    def user_utt(self):
        return self._user_utt
    
    @property
    def user_utt_as_text(self):
        return self._user_utt_as_text
    
    @property
    def user_utt_lengths(self):
        return self._user_utt_lengths

    @property
    def response_node_ids(self):
        return self._response_node_ids

    @property
    def flowchart_names(self):
        return self._flowchart_names
    
    @property
    def size(self):
        return len(self._response_node_ids)

    def _populate_data_from_json(self, dataJson, glob):

        flowchart_name = dataJson['flowchart']
        for dialog in dataJson['dialogs']:
            
            context_vector = []
            context_as_text = []
            context_lengths = []
            
            utterences = dialog['utterences']
            exchanges = [utterences[i * 2:(i + 1) * 2] for i in range((len(utterences) + 2 - 1) // 2 )]
            for exchange in exchanges:
                
                user_utt_as_text =  exchange[0]['utterance']
                user_utt_length, user_utt_vector = vectorize_text(user_utt_as_text, glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                
                response_node_id = exchange[1]['node']
                self._response_node_ids.append(response_node_id)
                
                self._flowchart_names.append(flowchart_name)

                self._user_utt.append(user_utt_vector)
                self._user_utt_as_text.append(user_utt_as_text)
                self._user_utt_lengths.append(user_utt_length)

                # add previous agent utterance
                if len(context_vector) > 0:
                    self._agent_utt.append(context_vector[-1])
                    self._agent_utt_as_text.append(context_as_text[-1])
                    self._agent_utt_lengths.append(context_lengths[-1])
                else:
                    zero_vector_length, zero_vector = vectorize_text( "", glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                    self._agent_utt.append(zero_vector)
                    self._agent_utt_as_text.append("")
                    self._agent_utt_lengths.append(zero_vector_length)

                agent_utt_as_text =  exchange[1]['utterance']
                agent_utt_length, agent_utt_vector = vectorize_text(agent_utt_as_text, glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                context_vector += [user_utt_vector, agent_utt_vector]
                context_as_text += [user_utt_as_text, agent_utt_as_text]
                context_lengths += [user_utt_length, agent_utt_length]

class SiameseBatch(object):

    def __init__(self, data, flowcharts, glob, start, end, test=False):
        
        if test:
            # in test batch size is set to be 1
            # all negative examples to be added for evaluation
            flowchart_name = data.flowchart_names[start]
            no_of_nodes_in_flowchart = len(flowcharts.get_all_node_ids(flowchart_name))
            repeats = no_of_nodes_in_flowchart
        else:
            repeats = 2
        
        self.user_utt = np.repeat(data.user_utt[start:end], repeats=repeats, axis=0)
        self.user_utt_lens = np.repeat(data.user_utt_lengths[start:end], repeats=repeats, axis=0)
        self.agent_utt = np.repeat(data.agent_utt[start:end], repeats=repeats, axis=0)
        self.agent_utt_lens = np.repeat(data.agent_utt_lengths[start:end], repeats=repeats, axis=0)
            
        self.flowchart_names = np.repeat(data.flowchart_names[start:end], repeats=repeats, axis=0)
        self.response_node_ids = np.repeat(data.response_node_ids[start:end], repeats=repeats, axis=0)
        
        # for each (user-utt,agent-utt) pair:
        #   pick the positive example and a random negative example based on the (flowchart, response_node_id) pair
        self.node_text = []
        self.node_text_lens = []
        self.edge_text = []
        self.edge_text_lens = []
        self.gt_labels = []

        self.printable_text = []

        if test:

            response_node_id = data.response_node_ids[start]
            name = data.flowchart_names[start]
            
            flowchart_node_ids = flowcharts.get_all_node_ids(name)
            for flowchart_node_id in flowchart_node_ids:
                
                node_text =  flowcharts.get_parent_node_text(name, flowchart_node_id)
                node_text_length, node_text_vector = vectorize_text(node_text, glob['encoder_vocab_to_idx'], flowcharts.max_node_utterance_length)
                self.node_text.append(node_text_vector)
                self.node_text_lens.append(node_text_length)
                
                edge_text =  flowcharts.get_edge_from_parent(name, flowchart_node_id)
                edge_text_length, edge_text_vector = vectorize_text(edge_text, glob['encoder_vocab_to_idx'], flowcharts.max_edge_utterance_length)
                self.edge_text.append(edge_text_vector)
                self.edge_text_lens.append(edge_text_length)
                
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
                    node_text_length, node_text_vector = vectorize_text(node_text, glob['encoder_vocab_to_idx'], flowcharts.max_node_utterance_length)
                    self.node_text.append(node_text_vector)
                    self.node_text_lens.append(node_text_length)
                    
                    edge_text =  flowcharts.get_edge_from_parent(name, response_node_id)
                    edge_text_length, edge_text_vector = vectorize_text(edge_text, glob['encoder_vocab_to_idx'], flowcharts.max_edge_utterance_length)
                    self.edge_text.append(edge_text_vector)
                    self.edge_text_lens.append(edge_text_length)

                    self.gt_labels.append(1)

                else: # add false label
                    
                    random_node_id = random.choice(list(flowcharts.get_all_node_ids(name)))
                    while random_node_id == response_node_id:
                        random_node_id = random.choice(list(flowcharts.get_all_node_ids(name)))

                    node_text =  flowcharts.get_parent_node_text(name, random_node_id)
                    node_text_length, node_text_vector = vectorize_text(node_text, glob['encoder_vocab_to_idx'], flowcharts.max_node_utterance_length)
                    self.node_text.append(node_text_vector)
                    self.node_text_lens.append(node_text_length)
                    
                    edge_text =  flowcharts.get_edge_from_parent(name, random_node_id)
                    edge_text_length, edge_text_vector = vectorize_text(edge_text, glob['encoder_vocab_to_idx'], flowcharts.max_edge_utterance_length)
                    self.edge_text.append(edge_text_vector)
                    self.edge_text_lens.append(edge_text_length)
                    
                    self.gt_labels.append(0)