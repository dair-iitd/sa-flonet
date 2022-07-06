import copy
import random
import numpy as np

from transformers import BertTokenizer

from utils import PAD, PAD_INDEX, UNK, UNK_INDEX, GO_SYMBOL, GO_SYMBOL_INDEX, EOS, EOS_INDEX, EMPTY_INDEX
from utils import vectorize_text

from utils import read_flowchart_jsons, read_dialog_jsons, build_vocab, create_batches

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Data(object):
    
    def __init__(self, dataJson, glob):

        self._context = []
        self._context_as_text = []
        self._context_lengths = []

        self._query = []
        self._query_as_text = []
        self._query_lengths = []

        self._response = []
        self._response_as_text = []
        self._response_lengths = []
        self._response_node_id = []
        self._flowchart_names = []

        self._agent_query_for_siamese = []
        self._agent_query_for_siamese_as_text = []
        self._agent_query_for_siamese_lengths = []

        self._user_query_for_siamese = []
        self._user_query_for_siamese_as_text = []
        self._user_query_for_siamese_lengths = []
        
        self._populate_data_from_json(dataJson, glob)

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

    @property
    def response_node_id(self):
        return self._response_node_id

    @property
    def flowchart_names(self):
        return self._flowchart_names
    
    def size(self):
        return len(self._response_node_id)
    
    @property
    def agent_query_for_siamese(self):
        return self._agent_query_for_siamese
    
    @property
    def agent_query_for_siamese_as_text(self):
        return self._agent_query_for_siamese_as_text
    
    @property
    def agent_query_for_siamese_lengths(self):
        return self._agent_query_for_siamese_lengths

    @property
    def user_query_for_siamese(self):
        return self._user_query_for_siamese
    
    @property
    def user_query_for_siamese_as_text(self):
        return self._user_query_for_siamese_as_text
    
    @property
    def user_query_for_siamese_lengths(self):
        return self._user_query_for_siamese_lengths

    def _populate_data_from_json(self, dataJson, glob):

        flowchart_name = dataJson['flowchart']
        for dialog in dataJson['dialogs']:
            
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

                response_node_id = exchange[1]['node']
                self._response_node_id.append(response_node_id)
                
                self._flowchart_names.append(flowchart_name)

                self._context.append(copy.deepcopy(context_vector))
                self._context_as_text.append(copy.deepcopy(context_as_text))
                self._context_lengths.append(copy.deepcopy(context_lengths))

                self._user_query_for_siamese.append(query_vector)
                self._user_query_for_siamese_as_text.append(query_as_text)
                self._user_query_for_siamese_lengths.append(query_vector_length)

                if len(context_vector) > 0:
                    self._agent_query_for_siamese.append(context_vector[-1])
                    self._agent_query_for_siamese_as_text.append(context_as_text[-1])
                    self._agent_query_for_siamese_lengths.append(context_lengths[-1])
                else:
                    zero_vector_length, zero_vector = vectorize_text( "", glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                    self._agent_query_for_siamese.append(zero_vector)
                    self._agent_query_for_siamese_as_text.append("")
                    self._agent_query_for_siamese_lengths.append(zero_vector_length)

                # response encoded using encoder vocab
                modified_response_vector_length, modified_response_vector = vectorize_text(response_as_text, glob['encoder_vocab_to_idx'], glob['max_input_sent_length'])
                context_vector += [query_vector, modified_response_vector]
                context_as_text += [query_as_text, response_as_text]
                context_lengths += [query_vector_length, modified_response_vector_length]

class Flowcharts(object):

    def __init__(self, flowcharts, glob):
        self._flowcharts = flowcharts
        self._max_node_utterance_length = glob['max_node_utterance_length']
        self._max_edge_utterance_length = glob['max_edge_label_length']
        self._encoder_vocab_to_idx = glob['encoder_vocab_to_idx']
        self._populate_node_properties()
        self._populate_node_representations()
        self._flowchart_to_stories()
    
    @property
    def node_text_for_siamese(self):
        return self._node_text_for_siamese
    
    @property
    def node_text_for_siamese_as_text(self):
        return self._node_text_for_siamese_as_text

    @property
    def node_text_for_siamese_lengths(self):
        return self._node_text_for_siamese_lengths

    @property
    def edge_text_for_siamese(self):
        return self._edge_text_for_siamese

    @property
    def edge_text_for_siamese_as_text(self):
        return self._edge_text_for_siamese_as_text

    @property
    def edge_text_for_siamese_lengths(self):
        return self._edge_text_for_siamese_lengths

    def _populate_node_properties(self):
        self._paths_to_root = {}
        self._node_to_text_map  = {}
        for name, flowchart in self._flowcharts.items():
            parent_map = self._get_parents_map(flowchart)
            name = flowchart['name']
            self._paths_to_root[name] = {}
            self._node_to_text_map[name] = {}
            for node_id, node_properties in flowchart['nodes'].items():
                path_to_root = []
                curr_node = node_id
                while (curr_node in parent_map):
                    path_to_root.insert(0, parent_map[curr_node])
                    curr_node = parent_map[curr_node][0]
                self._paths_to_root[name][node_id] = copy.deepcopy(path_to_root)
                self._node_to_text_map[name][node_id] = node_properties['utterance']

    def _get_parents_map(self, flowchart):
        parent_map = {}
        for parent_node_id, edges in flowchart['edges'].items():
            for option, child_node_id in edges.items():
                parent_map[child_node_id] = (parent_node_id, option)
        return parent_map
    
    def _populate_node_representations(self):
        
        self._node_text_for_siamese = {}
        self._node_text_for_siamese_as_text = {}
        self._node_text_for_siamese_lengths = {}

        self._edge_text_for_siamese = {}
        self._edge_text_for_siamese_as_text = {}
        self._edge_text_for_siamese_lengths = {}
        
        for name, flowchart in self._flowcharts.items():
            
            name = flowchart['name']
            self._node_text_for_siamese[name] = {}
            self._node_text_for_siamese_as_text[name] = {}
            self._node_text_for_siamese_lengths[name] = {}

            self._edge_text_for_siamese[name] = {}
            self._edge_text_for_siamese_as_text[name] = {}
            self._edge_text_for_siamese_lengths[name] = {}

            paths_to_root = self._paths_to_root[name]
            for node_id, node_properties in flowchart['nodes'].items():
                if len(paths_to_root[node_id]) == 0:
                    zero_vector_length, zero_vector = vectorize_text( "", self._encoder_vocab_to_idx, self._max_node_utterance_length)
                    self._node_text_for_siamese[name][node_id] = copy.deepcopy(zero_vector)
                    self._node_text_for_siamese_as_text[name][node_id] = ""
                    self._node_text_for_siamese_lengths[name][node_id] = zero_vector_length

                    zero_vector_length, zero_vector = vectorize_text( "", self._encoder_vocab_to_idx, self._max_edge_utterance_length)
                    self._edge_text_for_siamese[name][node_id] = copy.deepcopy(zero_vector)
                    self._edge_text_for_siamese_as_text[name][node_id] = ""
                    self._edge_text_for_siamese_lengths[name][node_id] = zero_vector_length
                else:
                    prev_node_edge_pair = paths_to_root[node_id][-1]

                    node_text = self._node_to_text_map[name][prev_node_edge_pair[0]]
                    node_text_length, node_text_vector = vectorize_text(node_text, self._encoder_vocab_to_idx, self._max_node_utterance_length)
                    self._node_text_for_siamese[name][node_id] = copy.deepcopy(node_text_vector)
                    self._node_text_for_siamese_as_text[name][node_id] = node_text
                    self._node_text_for_siamese_lengths[name][node_id] = node_text_length

                    edge_text = prev_node_edge_pair[1]
                    edge_text_length, edge_text_vector = vectorize_text(edge_text, self._encoder_vocab_to_idx, self._max_edge_utterance_length)
                    self._edge_text_for_siamese[name][node_id] = copy.deepcopy(edge_text_vector)
                    self._edge_text_for_siamese_as_text[name][node_id] = edge_text
                    self._edge_text_for_siamese_lengths[name][node_id] = edge_text_length                    

    def _flowchart_to_stories(self):
        self.stories=[]
        self.stories_as_text=[]
        self.story_lengths=[]
        self.max_story_length=0
        separator = EMPTY_INDEX#TODO can PAD be separator??
        #TODO Add flowchart names as well. Tokenised
        #TODO FIGURE OUT A WAY TO MAKE STORIES. CURRENT METHOD IS NOT GOOD
        #story = (node,separator,edge,separator,node)
        for flowchart_name, flowchart in self._flowcharts.items():
            for node, edges in flowchart['edges'].items():
                for edge_text, next_node in edges.items():
                    node_text_length, node_text_vector = vectorize_text(flowchart['nodes'][node]['utterance'], self._encoder_vocab_to_idx, self._max_node_utterance_length)
                    edge_text_length, edge_text_vector = vectorize_text(edge_text, self._encoder_vocab_to_idx, self._max_edge_utterance_length)
                    next_node_text_length, next_node_text_vector = vectorize_text(flowchart['nodes'][next_node]['utterance'], self._encoder_vocab_to_idx, self._max_node_utterance_length)
                    story=node_text_vector+[separator]+edge_text_vector+[separator]+next_node_text_vector
                    story_length=node_text_length+edge_text_length+next_node_text_length+2
                    story_as_text=[flowchart['nodes'][node]['utterance'],edge_text,flowchart['nodes'][next_node]['utterance']]
                    self.stories.append(story)
                    self.stories_as_text.append(story_as_text)
                    self.story_lengths.append(story_length)
                    self.max_story_length=story_length if story_length>self.max_story_length else self.max_story_length
        self.stories=np.array(self.stories)

class SiameseBatch(object):

    def __init__(self, data, flowcharts, start, end, loss='pairwise', test=False, bert=False):
        
        if test or loss == 'listwise':
            # in test batch size is set to be 1
            # all negative examples to be added for evaluation
            flowchart_name = data.flowchart_names[start]
            no_of_nodes_in_flowchart = len(flowcharts.node_text_for_siamese[flowchart_name].keys())
            repeats = no_of_nodes_in_flowchart
        else:
            repeats = 2

        user_utts = []
        agent_utts = []
        if bert:
            user_utts = np.repeat(data.user_query_for_siamese_as_text[start:end], repeats=repeats, axis=0)
            agent_utts = np.repeat(data.agent_query_for_siamese_as_text[start:end], repeats=repeats, axis=0)
        else:
            self.user_utt = np.repeat(data.user_query_for_siamese[start:end], repeats=repeats, axis=0)
            self.user_utt_lens = np.repeat(data.user_query_for_siamese_lengths[start:end], repeats=repeats, axis=0)
            self.agent_utt = np.repeat(data.agent_query_for_siamese[start:end], repeats=repeats, axis=0)
            self.agent_utt_lens = np.repeat(data.agent_query_for_siamese_lengths[start:end], repeats=repeats, axis=0)
            
        self.flowchart_names = np.repeat(data.flowchart_names[start:end], repeats=repeats, axis=0)
        self.response_node_id = np.repeat(data.response_node_id[start:end], repeats=repeats, axis=0)
        
        self.node_text = []
        self.node_text_lens = []
        self.edge_text = []
        self.edge_text_lens = []
        self.gt_labels = []

        self.printable_text = []

        node_texts = []
        edge_texts = []
            
        if test or loss == 'listwise':
            node_id = data.response_node_id[start]
            name = data.flowchart_names[start]

            for flowchart_node_id in flowcharts.node_text_for_siamese[name].keys():

                if bert == False:
                    self.node_text.append(copy.deepcopy(flowcharts.node_text_for_siamese[name][flowchart_node_id]))
                    self.node_text_lens.append(copy.deepcopy(flowcharts.node_text_for_siamese_lengths[name][flowchart_node_id]))
                    self.edge_text.append(copy.deepcopy(flowcharts.edge_text_for_siamese[name][flowchart_node_id]))
                    self.edge_text_lens.append(copy.deepcopy(flowcharts.edge_text_for_siamese_lengths[name][flowchart_node_id]))
                else:
                    node_texts.append(flowcharts.node_text_for_siamese_as_text[name][flowchart_node_id])
                    edge_texts.append(flowcharts.edge_text_for_siamese_as_text[name][flowchart_node_id])

                if flowchart_node_id == node_id:
                    self.gt_labels.append(1)
                else:
                    self.gt_labels.append(0)
            
                printable_text = {
                    'node_text': flowcharts.node_text_for_siamese_as_text[name][flowchart_node_id],
                    'edge_text': flowcharts.edge_text_for_siamese_as_text[name][flowchart_node_id],
                    'agent_utt': data.agent_query_for_siamese_as_text[start],
                    'user_utt' : data.user_query_for_siamese_as_text[start]
                }
                self.printable_text.append(copy.deepcopy(printable_text))
        else:
            for idx, node_id in enumerate(self.response_node_id):
                node_id = self.response_node_id[idx]
                name = self.flowchart_names[idx]

                if idx%2==0: # add true label
                    if bert == False:
                        self.node_text.append(copy.deepcopy(flowcharts.node_text_for_siamese[name][node_id]))
                        self.node_text_lens.append(copy.deepcopy(flowcharts.node_text_for_siamese_lengths[name][node_id]))
                        self.edge_text.append(copy.deepcopy(flowcharts.edge_text_for_siamese[name][node_id]))
                        self.edge_text_lens.append(copy.deepcopy(flowcharts.edge_text_for_siamese_lengths[name][node_id]))
                    else:
                        node_texts.append(flowcharts.node_text_for_siamese_as_text[name][node_id])
                        edge_texts.append(flowcharts.edge_text_for_siamese_as_text[name][node_id])
                    self.gt_labels.append(1)
                else: # add false label
                    random_node_id = random.choice(list(flowcharts.node_text_for_siamese[name].keys()))
                    while random_node_id == node_id:
                        random_node_id = random.choice(list(flowcharts.node_text_for_siamese[name].keys()))
                    if bert == False:
                        self.node_text.append(copy.deepcopy(flowcharts.node_text_for_siamese[name][random_node_id]))
                        self.node_text_lens.append(copy.deepcopy(flowcharts.node_text_for_siamese_lengths[name][random_node_id]))
                        self.edge_text.append(copy.deepcopy(flowcharts.edge_text_for_siamese[name][random_node_id]))
                        self.edge_text_lens.append(copy.deepcopy(flowcharts.edge_text_for_siamese_lengths[name][random_node_id]))
                    else:
                        node_texts.append(flowcharts.node_text_for_siamese_as_text[name][node_id])
                        edge_texts.append(flowcharts.edge_text_for_siamese_as_text[name][node_id])
                    self.gt_labels.append(0)
            
        if bert:
            self.user_utt, self.user_utt_lens, self.agent_utt_lens = self.tokenize_for_bert(user_utts, agent_utts, node_texts, edge_texts)
            self.agent_utt = []
            self.node_text, self.node_text_lens = [], []
            self.edge_text, self.edge_text_lens = [], []

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

class MemN2NBatch(object):
    def __init__(self,trnData,glob,start,end):
        max_input_sent_length=glob['max_input_sent_length']

        self.context=trnData.context[start:end]#(batch size,num_utterances,sentence length)
        self.context_num_utterances=list(map(lambda x:len(x),self.context))
        self.context_utterance_lengths=trnData.context_lengths[start:end]
        self.context_as_text=trnData.context_as_text[start:end]
        max_context_len=max(self.context_num_utterances)

        self.context=list(map(lambda x: np.array(x),self.context))
        self.context[self.context==[]]=np.ones((max_context_len,max_input_sent_length))*PAD_INDEX
        self.context=np.array(list(map(lambda x:np.concatenate((x,np.ones((max_context_len-x.shape[0],max_input_sent_length))*PAD_INDEX)),self.context)))
        self.context_utterance_lengths=np.array(list(map(lambda x:np.concatenate((x,np.zeros(max_context_len-len(x)))),self.context_utterance_lengths)))#(batch,max num_utterances)
        self.context_as_text=np.array(list(map(lambda x:np.array(x),self.context_as_text)))#(batch,num_utterances in the context)

        #queries
        self.query=np.array(trnData.query[start:end])#(batch,max_sentLen)
        self.query_as_text=np.array(trnData.query_as_text[start:end])#(batch,text)
        self.query_lengths=np.array(trnData.query_lengths[start:end])#(batch)

        #response
        self.response=np.array(trnData.response[start:end])#(batch,max_response_sentLen)
        self.response_as_text=np.array(trnData.response_as_text[start:end])#(batch,text)
        self.response_lengths=np.array(trnData.response_lengths[start:end])#(batch)

        self.flowchart_names=np.array(trnData.flowchart_names[start:end])#(batch)
        self.response_node_id=np.array(trnData.response_node_id[start:end])#(batch)

if __name__ == "__main__":
    
    args = {}
    args['flowchart-dir'] = "../data/flowcharts/"
    args['dialog-dir'] = "../data/dialogs/"

    flowcharts = read_flowchart_jsons(args['flowchart-dir'])
    trnJson, valJson, tstJson = read_dialog_jsons(args['dialog-dir'])
    glob = build_vocab(flowcharts, trnJson, valJson)

    trainData = Data(trnJson, glob)
    flowcharts = Flowcharts(flowcharts, glob)

    train_batches = create_batches(trainData, 1)
    for (start, end) in train_batches:
        batch = SiameseBatch(trainData, flowcharts, start, end, test=True)

