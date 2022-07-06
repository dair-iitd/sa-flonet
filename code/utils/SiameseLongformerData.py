import copy
import random
import numpy as np

from utils import vectorize_text
from transformers import LongformerTokenizer

longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
PAD_TOKEN_ID = longformer_tokenizer.pad_token_id

def longformerVectorize(text1,text2=''):
    if(text2==''):
        dialog_vector = longformer_tokenizer.encode(text1) 
    else:
        dialog_vector = longformer_tokenizer.encode(text1,text2) 
    return dialog_vector, len(dialog_vector)

def get_path_to_node_from_flowchart(flowcharts,flowchart_name,response_node_id):
    path_to_node=flowcharts.get_path_to_node(flowchart_name,response_node_id)
    path_as_text=['']
    for node,edge in path_to_node:
        path_as_text+=[flowcharts.get_node_text(flowchart_name,node),edge]
    return path_as_text

def get_path_to_node_from_history(flowcharts,flowchart_name,response_node_history):
    path_as_text = []
    for node in response_node_history:
        path_to_node+=[flowcharts.get_edge_from_parent(flowchart_name,node),node]
    #remove the last node from path
    if(len(path_as_text)>0):
        path_as_text.pop(-1)
    else:
        path_as_text=['']
    return path_as_text

class SiameseLongformerData(object):
    def __init__(self, dataJson, flowcharts, glob):
        self._context = []
        self._context_as_text = []
        self._context_lengths = []

        self._path = []
        self._path_as_text = []
        self._path_lengths = []

        self._response_node_ids = []
        self._path_flowchart = []
        self._flowchart_names = []
        self._max_context_length = 0
        self._max_path_length = 0
        self._populate_data_from_json(dataJson, flowcharts, glob)

    @property
    def context(self):
        return self._context
    
    @property
    def context_as_text(self):
        return self._context_as_text
    
    @property
    def context_lengths(self):
        return self._context_lengths

    @property
    def path(self):
        return self._path

    @property
    def path_as_text(self):
        return self._path_as_text
    
    @property
    def path_lengths(self):
        return self._path_lengths
    
    @property
    def response_node_ids(self):
        return self._response_node_ids

    @property
    def max_context_length(self):
        return self._max_context_length

    @property
    def max_path_length(self):
        return self._max_path_length

    @property
    def flowchart_names(self):
        return self._flowchart_names
    
    @property
    def path_flowchart(self):
        return self._path_flowchart
        
    @property
    def size(self):
        return len(self._response_node_ids)

    def _populate_data_from_json(self, dataJson, flowcharts, glob):

        flowchart_name = dataJson['flowchart']
        for dialog in dataJson['dialogs']:
            
            context_as_text = []
            path_as_text = []
            path_flowchart = []
            
            utterences = dialog['utterences']
            exchanges = [utterences[i * 2:(i + 1) * 2] for i in range((len(utterences) + 2 - 1) // 2 )]
            for exchange in exchanges:
                
                query_as_text =  exchange[0]['utterance']
                response_as_text = exchange[1]['utterance']
                response_node_id = exchange[1]['node']
                context_as_text += [query_as_text]
                context_vector, context_length = longformerVectorize(' '.join(context_as_text))
                self._context.append(copy.deepcopy(context_vector))
                self._context_as_text.append(copy.deepcopy(context_as_text))
                self._context_lengths.append(copy.deepcopy(context_length))
                self._response_node_ids.append(response_node_id)
                context_as_text += [response_as_text]

                self._flowchart_names.append(flowchart_name)
                '''
                path_as_text=get_path_to_node_from_flowchart(flowcharts,flowchart_name,response_node_id)
                '''
                node_text =  flowcharts.get_node_text(flowchart_name, response_node_id)
                edge_text =  flowcharts.get_edge_from_parent(flowchart_name, response_node_id)
                path_as_text += [edge_text]
                path_flowchart += [edge_text]
                #'''
                path_vector, path_length = longformerVectorize('. '.join(path_as_text)+".")
                self._path.append(copy.deepcopy(path_vector))
                self._path_as_text.append(copy.deepcopy(path_as_text))
                self._path_lengths.append(copy.deepcopy(path_length))
                self._path_flowchart.append(path_flowchart)
                path_as_text += [node_text]
                path_flowchart += [response_node_id]

        self._max_context_length = np.max(self._context_lengths)
        self._max_path_length = np.max(self._path_lengths)

class SiameseLongformerBatch(object):
    def __init__(self, data, flowcharts, glob, start, end, test=False):
        max_input_sent_length=data.max_context_length+data.max_path_length-1
        if test:
            # in test batch size is set to be 1
            # all negative examples to be added for evaluation
            flowchart_name = data.flowchart_names[start]
            no_of_nodes_in_flowchart = len(flowcharts.get_all_node_ids(flowchart_name))
            repeats = no_of_nodes_in_flowchart
        else:
            repeats = 2
        
        self.batch_sentids=[]
        self.batch_segids=[]
        self.batch_mask=[]
        self.gt_labels=[]

        context = np.repeat(data.context[start:end], repeats=repeats, axis=0)

        self.flowchart_names = np.repeat(data.flowchart_names[start:end], repeats=repeats, axis=0)
        self.response_node_ids = np.repeat(data.response_node_ids[start:end], repeats=repeats, axis=0)

        self.context_as_text = np.repeat(data.context_as_text[start:end], repeats=repeats, axis=0)
        self.path_as_text = np.repeat(data.path_as_text[start:end], repeats=repeats, axis=0)

        if(test):
            response_node_id = data.response_node_ids[start]
            name = data.flowchart_names[start]
            context=self.context_as_text[start]
            original_path=self.path_as_text[start]
            self.path_as_text=[]
            flowchart_node_ids = flowcharts.get_all_node_ids(name)
            for flowchart_node_id in flowchart_node_ids:
                text_path_to_node=get_path_to_node_from_flowchart(flowcharts,name,flowchart_node_id)
                if flowchart_node_id == response_node_id:
                    text_path_to_node=original_path
                joint,joint_length=longformerVectorize(" ".join(context),". ".join(text_path_to_node))
                _,context_length=longformerVectorize(" ".join(context))
                padding = list(np.ones(max_input_sent_length-joint_length)*PAD_TOKEN_ID)
                self.batch_sentids.append(joint+padding)
                self.batch_segids.append(list(np.zeros(context_length))+list(np.ones(joint_length-context_length))+list(np.zeros(len(padding))))
                self.batch_mask.append(list(np.ones(len(joint)))+list(np.zeros(len(padding))))
                self.path_as_text.append(text_path_to_node)
                if flowchart_node_id == response_node_id:
                    self.gt_labels.append(1)
                else:
                    self.gt_labels.append(0)
        else:
            for idx, response_node_id in enumerate(self.response_node_ids):
                if idx%2==0: # add true label
                    joint,joint_length=longformerVectorize(" ".join(self.context_as_text[idx]),". ".join(self.path_as_text[idx]))
                    _,context_length=longformerVectorize(" ".join(self.context_as_text[idx]))
                    padding = list(np.ones(max_input_sent_length-joint_length)*PAD_TOKEN_ID)
                    self.batch_sentids.append(joint+padding)
                    self.batch_segids.append(list(np.zeros(context_length))+list(np.ones(joint_length-context_length))+list(np.zeros(len(padding))))
                    self.batch_mask.append(list(np.ones(len(joint)))+list(np.zeros(len(padding))))
                    self.gt_labels.append(1)

                else: # add false label
                    #random path
                    flowchart = self.flowchart_names[idx]
                    random_node_id = random.choice(list(flowcharts.get_all_node_ids(flowchart)))
                    while random_node_id == response_node_id:
                        random_node_id = random.choice(list(flowcharts.get_all_node_ids(flowchart)))
                    random_path=get_path_to_node_from_flowchart(flowcharts,flowchart,random_node_id)
                    joint,joint_length=longformerVectorize(" ".join(self.context_as_text[idx]),". ".join(random_path))
                    _,context_length=longformerVectorize(" ".join(self.context_as_text[idx]))
                    padding = list(np.ones(max_input_sent_length-joint_length)*PAD_TOKEN_ID)
                    self.batch_sentids.append(joint+padding)
                    self.batch_segids.append(list(np.zeros(context_length))+list(np.ones(joint_length-context_length))+list(np.zeros(len(padding))))
                    self.batch_mask.append(list(np.ones(len(joint)))+list(np.zeros(len(padding))))
                    self.path_as_text[idx]=random_path
                    self.gt_labels.append(0)

        self.batch_sentids=np.array(self.batch_sentids)
        self.batch_segids=np.array(self.batch_segids)
        self.batch_mask=np.array(self.batch_mask)
        unused_length=min(np.sum(self.batch_mask==0,axis=1))
        useful_length=self.batch_mask.shape[1]-unused_length
        self.batch_sentids=self.batch_sentids[:,:useful_length]
        self.batch_segids=self.batch_segids[:,:useful_length]
        self.batch_mask=self.batch_mask[:,:useful_length]

