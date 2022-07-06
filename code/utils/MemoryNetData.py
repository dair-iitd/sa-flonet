import copy
import random
import numpy as np

from utils import PAD, PAD_INDEX, UNK, UNK_INDEX, GO_SYMBOL, GO_SYMBOL_INDEX, EOS, EOS_INDEX, EMPTY_INDEX, SEPARATOR, SEPARATOR_INDEX
from utils import vectorize_text
#from .SiameseData import SiameseData

class MemoryNetData(object):
    def __init__(self, dataJson, flowcharts, glob):
        self._context = []
        self._context_as_text = []
        self._context_lengths = []

        self._query = []
        self._query_as_text = []
        self._query_lengths = []

        self._response = []
        self._response_as_text = []
        self._response_lengths = []
        #self._response_node_id = []
        self._flowchart_names = []

        self._populate_data_from_json(dataJson, glob)
        self._flowchart_to_stories(flowcharts, glob)

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
    def query(self):
        return self._query

    @property
    def query_as_text(self):
        return self._query_as_text
    
    @property
    def query_lengths(self):
        return self._query_lengths

    @property
    def response(self):
        return self._response

    @property
    def response_as_text(self):
        return self._response_as_text
    
    @property
    def response_lengths(self):
        return self._response_lengths

    #@property
    #def response_node_id(self):
    #    return self._response_node_id

    @property
    def flowchart_names(self):
        return self._flowchart_names
    
    @property
    def size(self):
        return len(self._response)

    def _populate_data_from_json(self, dataJson, glob):

        for dialog in dataJson['dialogs']:
            flowchart_name = dialog['flowchart']
            context_vector = []
            context_as_text = []
            context_lengths = []
            
            utterences = dialog['utterences']
            exchanges = [utterences[i * 2:(i + 1) * 2] for i in range((len(utterences) + 2 - 1) // 2 )]
            for exchange in exchanges:
                
                query_as_text =  exchange[0]['utterance']
                query_vector_length, query_vector = vectorize_text(query_as_text, glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                self._query.append(query_vector)
                self._query_as_text.append(query_as_text)
                self._query_lengths.append(query_vector_length)

                response_as_text = exchange[1]['utterance']
                response_vector_length, response_vector = vectorize_text(response_as_text, glob['decoder_vocab_to_idx'], glob['max_response_sent_length'])
                self._response.append(response_vector)
                self._response_as_text.append(response_as_text)
                self._response_lengths.append(response_vector_length)

                #response_node_id = exchange[1]['node']
                #self._response_node_id.append(response_node_id)
                
                self._flowchart_names.append(flowchart_name)

                self._context.append(copy.deepcopy(context_vector))
                self._context_as_text.append(copy.deepcopy(context_as_text))
                self._context_lengths.append(copy.deepcopy(context_lengths))

                # response encoded using encoder vocab
                modified_response_vector_length, modified_response_vector = vectorize_text(response_as_text, glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                context_vector += [query_vector, modified_response_vector]
                context_as_text += [query_as_text, response_as_text]
                context_lengths += [query_vector_length, modified_response_vector_length]

    def _flowchart_to_stories(self, flowcharts, glob):
        self.flowchart_stories_map={}
        self.flowchart_stories_as_text_map={}
        self.flowchart_story_lengths_map={}
        self.max_story_length=0
        self.max_n_stories=0

        flowchartNames = self._flowchart_names
        full_flowcharts = [flowcharts.get_full_flowchart(name) for name in flowchartNames]
        encoder_vocab = glob['encoder_vocab_to_idx']
        separator = SEPARATOR_INDEX#TODO use a new separator
        #story = (node,separator,edge,separator,node)
        for flowchart_name, flowchart in dict(zip(flowchartNames,full_flowcharts)).items():
            stories = []
            stories_as_text = []
            story_lengths = []
            for node, edges in flowchart['edges'].items():
                for edge_text, next_node in edges.items():
                    node_text_length, node_text_vector = vectorize_text(flowchart['nodes'][node]['utterance'], encoder_vocab, flowcharts.max_node_utterance_length)
                    edge_text_length, edge_text_vector = vectorize_text(edge_text, encoder_vocab, flowcharts.max_edge_utterance_length)
                    next_node_text_length, next_node_text_vector = vectorize_text(flowchart['nodes'][next_node]['utterance'], encoder_vocab, flowcharts.max_node_utterance_length)
                    story=node_text_vector[:node_text_length]+[separator]+edge_text_vector[:edge_text_length]+[separator]+next_node_text_vector[:next_node_text_length]
                    story_length=node_text_length+edge_text_length+next_node_text_length+2
                    story_padding = len(node_text_vector)+len(edge_text_vector)+len(next_node_text_vector)+2-story_length
                    story = story + [0]*story_padding
                    story_as_text=[flowchart['nodes'][node]['utterance'],edge_text,flowchart['nodes'][next_node]['utterance']]
                    stories.append(story)
                    stories_as_text.append(story_as_text)
                    story_lengths.append(story_length)
                    self.max_story_length=max(story_length,self.max_story_length)
            self.flowchart_stories_map[flowchart_name]=np.array(stories)
            self.flowchart_stories_as_text_map[flowchart_name]=np.array(stories_as_text)
            self.flowchart_story_lengths_map[flowchart_name]=np.array(story_lengths)
            self.max_n_stories=max(len(stories),self.max_n_stories)

class MemN2NBatch(object):
    def __init__(self, data, glob, start, end):
        max_input_sent_length=glob['max_input_sent_length']

        self.context=data.context[start:end]#(batch size,num_utterances,sentence length)
        self.context_num_utterances=list(map(lambda x:len(x),self.context))
        self.context_utterance_lengths=data.context_lengths[start:end]
        #self.context_as_text=data.context_as_text[start:end]
        max_context_len=max(self.context_num_utterances)

        self.context=list(map(lambda x: np.array(x),self.context))
        default = np.ones((max_context_len,max_input_sent_length))*PAD_INDEX
        self.context=list(map(lambda x:default if len(x)== 0 else x,self.context))
        self.context=np.array(list(map(lambda x:np.concatenate((x,np.ones((max_context_len-x.shape[0],max_input_sent_length))*PAD_INDEX)),self.context)))
        self.context_utterance_lengths=np.array(list(map(lambda x:np.concatenate((x,np.zeros(max_context_len-len(x)))),self.context_utterance_lengths)))#(batch,max num_utterances)
        #self.context_as_text=np.array(list(map(lambda x:np.array(x),self.context_as_text)))#(batch,num_utterances in the context)

        #queries
        self.query=np.array(data.query[start:end])#(batch,max_sentLen)
        #self.query_as_text=np.array(data.query_as_text[start:end])#(batch,text)
        self.query_lengths=np.array(data.query_lengths[start:end])#(batch)

        #response
        self.response=np.array(data.response[start:end])#(batch,max_response_sentLen)
        #self.response_as_text=np.array(data.response_as_text[start:end])#(batch,text)
        self.response_lengths=np.array(data.response_lengths[start:end])#(batch)

        self.flowchart_names=np.array(data.flowchart_names[start:end])#(batch)
        #self.response_node_id=np.array(data.response_node_id[start:end])#(batch)

        #stories
        self.batch_stories( data, data.flowchart_names[start:end])

    def batch_stories(self, data, flowchart_list):
        self.stories=[]
        self.stories_as_text=[]
        self.story_lengths=[]
        self.stories = [np.array(data.flowchart_stories_map[name]) for name in flowchart_list]
        self.stories = np.array([np.concatenate((x,np.array([[0]*x.shape[-1]]*(data.max_n_stories-x.shape[0])))) if data.max_n_stories!=x.shape[0] else x for x in self.stories])
        #self.stories_as_text = np.array([np.array(data.flowchart_stories_as_text_map[name]) for name in flowchart_list])
        self.story_lengths = np.array([np.array(list(data.flowchart_story_lengths_map[name])+[0]*(data.max_n_stories-len(data.flowchart_story_lengths_map[name]))) for name in flowchart_list])
