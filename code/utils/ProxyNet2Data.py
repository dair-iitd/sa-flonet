import copy
import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from utils import PAD, PAD_INDEX, UNK, UNK_INDEX, GO_SYMBOL, GO_SYMBOL_INDEX, EOS, EOS_INDEX, EMPTY_INDEX, SEPARATOR, SEPARATOR_INDEX
from utils import vectorize_text
#from .SiameseData import SiameseData

class ProxyNet2Data(object):
    def __init__(self, dataJson, flowcharts, glob, test):
        self._context = []
        self._context_as_text = []
        self._context_lengths = []

        self._response = []
        self._response_as_text = []
        self._response_lengths = []
        self._flowchart_names = []

        self.max_chart_utterance_length = 0

        self._label = []
        self.gt_label = []

        self._process_flowcharts_for_paths(flowcharts, glob)
        self._populate_data_from_json(dataJson, glob, test, flowcharts)

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
    def response(self):
        return self._response

    @property
    def response_as_text(self):
        return self._response_as_text
    
    @property
    def response_lengths(self):
        return self._response_lengths

    @property
    def flowchart_names(self):
        return self._flowchart_names

    @property
    def label(self):
        return self._label    
    @property
    def size(self):
        return len(self._label)

    def _populate_data_from_json(self, dataJson, glob, test, flowcharts):

        self.chart_data_array = list(self.chart_data_dict.values())
        self.chart_data_keys = list(self.chart_data_dict.keys())
        self.chart_data_idxs = []
        for dialog in dataJson['dialogs']:
            flowchart_name = dialog['flowchart']
            chart_data_idx = self.chart_data_keys.index(flowchart_name)
            context_vector = []
            context_as_text = []
            context_lengths = []
            
            utterences = dialog['utterences']
            exchanges = [utterences[i * 2:(i + 1) * 2] for i in range((len(utterences) + 2 - 1) // 2 )]
            for exchange in exchanges:
                #dialog context
                query_as_text =  exchange[0]['utterance']
                query_vector_length, query_vector = vectorize_text(query_as_text, glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                context_vector += [query_vector]
                context_as_text += [query_as_text]
                context_lengths += [query_vector_length]

                #dialog response
                response_as_text = exchange[1]['utterance']
                response_vector_length, response_vector = vectorize_text(response_as_text, glob['decoder_vocab_to_idx'], glob['max_response_sent_length'])

                self._context.append(copy.deepcopy(context_vector))
                self._context_as_text.append(copy.deepcopy(context_as_text))
                self._context_lengths.append(copy.deepcopy(context_lengths))
                self._response.append(response_vector)
                self._response_as_text.append(response_as_text)
                self._response_lengths.append(response_vector_length)

                #ground truth response
                gt_response = flowcharts.get_node_text(flowchart_name,exchange[1]['node'])

                #flowchart context and response by idx
                sorted_score_indexes = self._get_response_scores(response_as_text, self.chart_data_array[chart_data_idx]['chart_response_as_text'])
                chart_idx = sorted_score_indexes[0]

                self.chart_data_idxs.append(chart_data_idx)
                self._label.append(chart_idx)
                self.gt_label.append(list(self.chart_data_array[chart_data_idx]['chart_response_as_text']).index(gt_response))
                self._flowchart_names.append(flowchart_name)

                # response encoded using encoder vocab
                modified_response_vector_length, modified_response_vector = vectorize_text(response_as_text, glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                context_vector += [modified_response_vector]
                context_as_text += [response_as_text]
                context_lengths += [modified_response_vector_length]

    def _process_flowcharts_for_paths(self,flowcharts, glob):
        self.chart_data_dict = {}
        encoder_vocab = glob['encoder_vocab_to_idx']
        max_utterance_length = max(flowcharts.max_node_utterance_length,flowcharts.max_edge_utterance_length)
        self.max_chart_utterance_length = max_utterance_length
        for chart in flowcharts.get_flowchart_names():
            chart_paths=[]
            chart_responses=[]
            chart_path_lengths=[]
            chart_responses_length=[]            
            chart_paths_as_text=[]
            chart_responses_as_text=[]
            for node in list(flowcharts.get_all_node_ids(chart)):
                path = flowcharts.get_path_to_node(chart,node)
                path_text = []
                path_vector = []
                path_lengths = []
                for n, e in path:
                    node_text_length, node_text_vector = vectorize_text(flowcharts.get_node_text(chart,n), encoder_vocab, max_utterance_length)
                    edge_text_length, edge_text_vector = vectorize_text(e, encoder_vocab, max_utterance_length)

                    path_text+=[flowcharts.get_node_text(chart,n),e]
                    path_vector+=[node_text_vector,edge_text_vector]
                    path_lengths+=[node_text_length,edge_text_length]
                
                response_text_length, response_text_vector = vectorize_text(flowcharts.get_node_text(chart,node), encoder_vocab, flowcharts.max_node_utterance_length)

                chart_paths.append(path_vector)
                chart_responses.append(response_text_vector)
                chart_path_lengths.append(path_lengths)
                chart_responses_length.append(response_text_length)
                chart_paths_as_text.append(path_text)
                chart_responses_as_text.append(flowcharts.get_node_text(chart,node))
                    
            self.chart_data_dict[chart]={'path':chart_paths,
                                    'chart_response':chart_responses,
                                    'path_lengths':chart_path_lengths,
                                    'chart_response_lengths':chart_responses_length,
                                    'path_as_text':chart_paths_as_text,
                                    'chart_response_as_text':chart_responses_as_text}

    def _get_response_scores(self, text,chart_responses):
        bleu = []
        for _, response in enumerate(chart_responses):
            bleu.append(sentence_bleu([text.split()],response.split()))
        return np.argsort(np.array(bleu))[::-1]

    def batchify_charts(self):
        self.batchified_charts_data = []
        for _, chart_data in enumerate(self.chart_data_array):
            #chart responses
            chart_data['chart_response']=np.array(chart_data['chart_response'])#(batch,max_response_sentLen)
            chart_data['chart_response_lengths']=np.array(chart_data['chart_response_lengths'])#(batch)

            #flowchart path
            chart_data['path'], chart_data['path_num_utterances'], chart_data['path_utterance_lengths'] = self._batchify_contexts( chart_data['path'], chart_data['path_lengths'], self.max_chart_utterance_length, 0, len(chart_data['path_lengths']))

            self.batchified_charts_data.append(objectview(chart_data))
    
    def _batchify_contexts(self, contexts, context_lengths, max_len, start, end):
        batched_contexts = contexts[start:end]#(batch size,num_utterances,sentence length)
        context_num_utterances=list(map(lambda x:len(x),batched_contexts))
        max_context_len=max(context_num_utterances)
        context_utterance_lengths= context_lengths[start:end]

        #make a matrix from context
        max_input_sent_length=max_len
        batched_contexts=list(map(lambda x: np.array(x),batched_contexts))
        default = np.ones((max_context_len,max_input_sent_length))*PAD_INDEX
        batched_contexts=list(map(lambda x:default if len(x)== 0 else x,batched_contexts))
        batched_contexts=np.array(list(map(lambda x:np.concatenate((x,np.ones((max_context_len-x.shape[0],max_input_sent_length))*PAD_INDEX)),batched_contexts)))

        context_utterance_lengths=np.array(list(map(lambda x:np.concatenate((x,np.zeros(max_context_len-len(x)))),context_utterance_lengths)))#(batch,max num_utterances)
        return batched_contexts, context_num_utterances, context_utterance_lengths

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class ProxyNet2Batch(object):
    def __init__(self, data, glob, start, end):

        #context
        self.context, self.context_num_utterances, self.context_utterance_lengths = self._batchify_contexts( data.context, data.context_lengths, glob['max_input_sent_length'], start, end)
        self.context_as_text=data.context_as_text[start:end]

        #response
        self.response=np.array(data.response[start:end])#(batch,max_response_sentLen)
        self.response_as_text=np.array(data.response_as_text[start:end])#(batch,text)
        self.response_lengths=np.array(data.response_lengths[start:end])#(batch)

        self.flowchart_names=np.array(data.flowchart_names[start:end])#(batch)
        self.label=np.array(data.label[start:end])#(batch)
        self.chart_data_idxs=np.array(data.chart_data_idxs[start:end])#(batch)
        if 'gt_label' in data.__dict__:
            self.gt_label=np.array(data.gt_label[start:end])#(batch)
        assert len(list(set(self.flowchart_names)))==1#works only when batch has same flowchart
        self.chart_data = data.batchified_charts_data[self.chart_data_idxs[0]]

    def _batchify_contexts(self, contexts, context_lengths, max_len, start, end):
        batched_contexts = contexts[start:end]#(batch size,num_utterances,sentence length)
        context_num_utterances=list(map(lambda x:len(x),batched_contexts))
        max_context_len=max(context_num_utterances)
        context_utterance_lengths= context_lengths[start:end]

        #make a matrix from context
        max_input_sent_length=max_len
        batched_contexts=list(map(lambda x: np.array(x),batched_contexts))
        default = np.ones((max_context_len,max_input_sent_length))*PAD_INDEX
        batched_contexts=list(map(lambda x:default if len(x)== 0 else x,batched_contexts))
        batched_contexts=np.array(list(map(lambda x:np.concatenate((x,np.ones((max_context_len-x.shape[0],max_input_sent_length))*PAD_INDEX)),batched_contexts)))

        context_utterance_lengths=np.array(list(map(lambda x:np.concatenate((x,np.zeros(max_context_len-len(x)))),context_utterance_lengths)))#(batch,max num_utterances)
        return batched_contexts, context_num_utterances, context_utterance_lengths
