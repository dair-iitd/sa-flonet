import copy
import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from utils import PAD, PAD_INDEX, UNK, UNK_INDEX, GO_SYMBOL, GO_SYMBOL_INDEX, EOS, EOS_INDEX, EMPTY_INDEX, SEPARATOR, SEPARATOR_INDEX
from utils import vectorize_text
#from .SiameseData import SiameseData
#from .proxy_scoring_utils import PATH_SEPARATOR

class ProxyScoreOnlyFloData(object):
    def __init__(self, dataJson, flowcharts, glob, score_dict, flowchartDocsJson,test=False):
        self._context = []
        self._context_as_text = []
        self._context_lengths = []

        self._response = []
        self._response_as_text = []
        self._response_lengths = []
        self._flowchart_names = []

        self.dialog_data_index = []#index into context and response array

        self.path = []
        self.path_as_text = []
        self.path_lengths = []

        self.chart_response = []
        self.chart_response_as_text = []
        self.chart_response_lengths = []
        self.max_chart_utterance_length = 0

        self._label = []
        self.score = []
        self.gt_label = []
        self.batche_start = []
        
        self.last_node = []

        self._process_flowcharts_for_paths(flowcharts, glob, flowchartDocsJson)
        self.flowchart_doc_q2node(flowchartDocsJson)
        self.index_faqs(flowchartDocsJson)
        self._populate_data_from_json(dataJson, glob, flowcharts, score_dict,test=test)

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
        return len(self._response)

    def _populate_data_from_json(self, dataJson, glob, flowcharts, score_dict,test=False):

        for dialog in dataJson['dialogs']:
            flowchart_name = dialog['flowchart']
            chart_data = self.chart_data_dict[flowchart_name]
            scores = score_dict[flowchart_name]
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

                #ground truth response
                gt_response = "FAQ_NODE" if exchange[1]['node'] == None else flowcharts.get_node_text(flowchart_name,exchange[1]['node'])
                if 'FinalNode' in exchange[1] and 'FAQ:' in exchange[1]['FinalNode']:
                    gt_response = "FAQ_NODE"

                #######chart info
                #flowchart context and response by idx
                #PATH_SEPARATOR.join(context_as_text)
                paths_with_scores = [(x['Retrieved Contexts'],x['Scores']) for x in scores if x['Context']==context_as_text]
                assert len(paths_with_scores)>=1
                if len(paths_with_scores)>1:##temp
                    paths_with_scores = [paths_with_scores[0]]##temp

                chart_paths, chart_scores = paths_with_scores[0]
                score_map = dict(zip([" ".join(x) for x in chart_paths], chart_scores))

                self._context.append(copy.deepcopy(context_vector))
                self._context_as_text.append(copy.deepcopy(context_as_text))
                self._context_lengths.append(copy.deepcopy(context_lengths))
                self._response.append(response_vector)
                self._response_as_text.append(response_as_text)
                self._response_lengths.append(response_vector_length)
                self.last_node.append(exchange==exchanges[-1])


                ##get max score
                max_score = 0
                for chart_idx, _path in enumerate(chart_data['paths_text']):
                    max_score = max(score_map[" ".join(_path)],max_score)

                found_max = False
                for chart_idx, _path in enumerate(chart_data['paths_text']):
                    score = score_map[" ".join(_path)]
                    self.dialog_data_index.append(len(self._context)-1)
                    #add a check to make sure sizes are same
                    if score==max_score and not found_max:
                        self._label.append(int(score==max_score))#mas score
                        found_max = True
                    else:
                        self._label.append(int(False))
                    self.score.append(score)
                    self._flowchart_names.append(flowchart_name)
                    self.batche_start.append(int(chart_idx==0))
                    if chart_data['paths_text'][chart_idx][0] in self.doc_q2node[flowchart_name]:
                        print("shouldn't be here")
                        break
                        node_ = self.doc_q2node[flowchart_name][chart_data['paths_text'][chart_idx][0]]
                        node_chart_response = flowcharts.get_node_text(flowchart_name,node_)
                        self.gt_label.append(int(gt_response==node_chart_response))
                    else:
                        self.gt_label.append(int(gt_response==chart_data['responses_text'][chart_idx]))
                    #self._label[-1]=self.gt_label[-1]##Remove this
                    #self.score[-1]=self.gt_label[-1]##Remove this
                #######

                # response encoded using encoder vocab
                modified_response_vector_length, modified_response_vector = vectorize_text(response_as_text, glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                context_vector += [modified_response_vector]
                context_as_text += [response_as_text]
                context_lengths += [modified_response_vector_length]

    def _process_flowcharts_for_paths(self, flowcharts, glob, flowchartDocsJson):
        self.chart_data_dict = {}
        encoder_vocab = glob['encoder_vocab_to_idx']
        max_utterance_length = max(flowcharts.max_node_utterance_length,flowcharts.max_edge_utterance_length, glob['max_input_sent_length'])
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
                if len(path)==0:
                    node_text_length, node_text_vector = vectorize_text("", encoder_vocab, max_utterance_length)
                    path_text+=[""]
                    path_vector+=[node_text_vector]
                    path_lengths+=[node_text_length]
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
                    
            #aDD DOCUMENTS
            '''
            for d in flowchartDocsJson[chart]['supporting_faqs']:
                path_lengths, path_vector = vectorize_text(d['q'], encoder_vocab, max_utterance_length)
                response_text_length, response_text_vector = vectorize_text(d['a'], encoder_vocab, max_utterance_length)
                chart_paths.append([path_vector])
                chart_responses.append(response_text_vector)
                chart_path_lengths.append([path_lengths])
                chart_responses_length.append(response_text_length)
                chart_paths_as_text.append([d['q']])
                chart_responses_as_text.append(d['a'])
            '''
            self.chart_data_dict[chart]={'paths':chart_paths,
                                    'responses':chart_responses,
                                    'paths_length':chart_path_lengths,
                                    'responses_length':chart_responses_length,
                                    'paths_text':chart_paths_as_text,
                                    'responses_text':chart_responses_as_text}

    def _get_response_scores(self, text,chart_responses):
        bleu = []
        for _, response in enumerate(chart_responses):
            bleu.append(sentence_bleu([text.split()],response.split()))
        return np.argsort(np.array(bleu))[::-1]

    def flowchart_doc_q2node(self, flowchartDocsJson):
        doc_q2node = {}
        for chart, json in flowchartDocsJson.items():
            chart_q2node = {}
            for d in json['supporting_faqs']:
                chart_q2node[d['q']]=d['node']
            doc_q2node[chart]=chart_q2node
        self.doc_q2node=doc_q2node
    
    def index_faqs(self, flowchartDocsJson):
        indexed_faqs = {}
        for chart, json in flowchartDocsJson.items():
            indexed_faq = {v['ID']:v for v in json['supporting_faqs']}
            indexed_faqs[chart]=indexed_faq
        self.indexed_faqs=indexed_faqs

class ProxyScoreOnlyFloBatch(object):
    def __init__(self, data, glob, start, end):

        ##fetch context and response info from the direct thingy
        #sanity check
        assert len(list(set(data.dialog_data_index[start:end])))==1
        idx = data.dialog_data_index[start]
        dialog_start = idx
        dialog_end = idx+1

        #now get chart data
        assert len(list(set(data.flowchart_names[start:end])))==1
        flowchart_name = data.flowchart_names[start]
        chart_data = data.chart_data_dict[flowchart_name]
        assert end-start == len(chart_data['paths'])

        #context
        self.context, self.context_num_utterances, self.context_utterance_lengths = self._batchify_contexts( data.context, data.context_lengths, glob['max_input_sent_length'], dialog_start, dialog_end)
        self.context_as_text=data.context_as_text[dialog_start:dialog_end]

        #flowchart path
        self.path, self.path_num_utterances, self.path_utterance_lengths = self._batchify_contexts( chart_data['paths'], chart_data['paths_length'], max(data.max_chart_utterance_length,glob['max_input_sent_length']), 0, len(chart_data['paths']))
        self.path_as_text=chart_data['paths_text']

        #response
        self.response=np.array(data.response[dialog_start:dialog_end])#(batch,max_response_sentLen)
        self.response_as_text=np.array(data.response_as_text[dialog_start:dialog_end])#(batch,text)
        self.response_lengths=np.array(data.response_lengths[dialog_start:dialog_end])#(batch)
        
        #chart response
        self.chart_response=np.array(chart_data['responses'])#(batch,max_response_sentLen)
        self.chart_response_as_text=np.array(chart_data['responses_text'])#(batch,text)
        self.chart_response_lengths=np.array(chart_data['responses_length'])#(batch)

        self.flowchart_names=np.array(data.flowchart_names[start:end])#(batch)
        self.label=np.array(data.label[start:end])#(batch)
        self.score=np.array(data.score[start:end])#(batch)
        if 'last_node' in data.__dict__:
            self.last_node = np.array(data.last_node[dialog_start:dialog_end])
        if 'gt_label' in data.__dict__:
            self.gt_label=np.array(data.gt_label[start:end])#(batch)

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
