from collections import Counter
from itertools import combinations
from math import sqrt
import matplotlib.pyplot as plt
import networkx as nx
from nltk import word_tokenize, sent_tokenize, FreqDist,pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter
import re

CONVERGENCE_THRESHOLD = 0.0001
NOUNS = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
class Document():
    '''
    The master class for our Document Summerization module.
    Incorporates all features related to Document
    '''
    
    def __init__(self, document):
        self.document = document
        self.sents = self.document.split('.')
        self.data = ['.'.join(self.sents[i:i+3]) for i in range(0,len(self.sents),3)]
        self.word_freq = FreqDist(clean(self.document))
        self.imp_sents = tf_idf(self.data,self.word_freq)
        self.graph = None
        self.params = { 'thresh': 0.0
            
        }
                
    def __str__(self):
        return self.document
    
    
    def statistical_sim(self, sent1, sent2):
        '''
        Statistical similarity between sentences
        based on the cosine method
        Returns: float (the cosine similarity b/w sent1 and sent2)
        '''
        sent_token1 = Counter(sent1)
        sent_token2 = Counter(sent2)
        
        intxn = set(sent_token1) & set(sent_token2)
        numerator = sum([sent_token1[x] * sent_token2[x] for x in intxn])
        
        mod1 = sum([sent_token1[x]**2 for x in sent_token1.keys()])
        mod2 = sum([sent_token2[x]**2 for x in sent_token2.keys()])
        denominator = sqrt(mod1)*sqrt(mod2)
        
        if not denominator:
            return 0.0

        return float(numerator)/denominator
    
    
    def semantic_sim(self, sent1, sent2):
        '''
        A semantic similarity score between two sentences
        based on WordNet
        Returns: float (the semantic similarity measure)
        '''
        score = 0
        sent1 = [word for word in sent1 if word in NOUNS]
        sent2 = [word for word in sent2 if word in NOUNS]
        for t1 in sent1:
            for t2 in sent2:
                score += semantic_score(t1,t2)
        try:
            return score/(len(sent1 + sent2))  
        except:
            return 10000
    
    
    def construct_graph(self):
        '''
        Constructs the word similarity graph
        '''
        connected = []
        for pair in combinations(self.imp_sents, 2):
            cpair = clean(pair[0]), clean(pair[1])
            weight = self.statistical_sim(*cpair) + self.semantic_sim(*cpair)
            connected.append((pair[0], pair[1], weight))
        self.graph = draw_graph(connected, self.params['thresh'])    

def clean(sent):
    '''
    A utility function that returns a a list of words in a sentence
    after cleaning it. Gets rid off uppper-case, punctuations, 
    stop words, etc.
    Returns: list (a list of cleaned words in sentence)
    '''
    words =  sent.lower() 
    words = re.findall(r'\w+', words,flags = re.UNICODE | re.LOCALE) 
    imp_words = filter(lambda x: x not in stopwords.words('english'), words)
    return imp_words
        
def semantic_score(word1, word2):
    '''
    Semantic score between two words based on WordNet
    Returns: float (the semantic score between word1 and word2)
    '''
    try:
        w1 = wn.synset('%s.n.01'%(word1))
        w2 = wn.synset('%s.n.01'%(word2))
        return wn.path_similarity(w1,w2,simulate_root = False)
    except:
        return 0
    
def draw_graph(connected, thresh):
    '''
    Draws graph as per weights and puts edges if 
    weight exceed the given thresh
    Returns: networkx Graph (nodes are sentences and edges
             are statistical and semantic relationships)
    '''
    nodes = set([n1 for n1, n2, n3 in connected] + [n2 for n1, n2, n3 in connected])
    G=nx.Graph()
    for node in nodes:
        G.add_node(node)
    for edge in connected:
        if edge[2] > thresh:
            G.add_edge(edge[0], edge[1],weight = edge[2])
    return G
def tf_idf(data,freq_dist):
    '''Rate each sentence of the text based on TF/IDF 
       rating mechanism. Text is a string and each
       sentence in the text is assumed to be separated
       by a dot'''
    scores = {}
    #freq_dist = FreqDist(filter_words)
    sents = filter(lambda a: len(a) > 0,data)
    for sent in sents:
        score = 0
        for word in sent.split(' '):
            word = word.strip('.')
            score += freq_dist[word]
        scores[sent] = float(score)/len(sent.split(' '))
    filtered = [i[0] for i in Counter(scores).most_common(20)]
    return filtered
    
def textrank_weighted(obj, initial_value=None, damping=0.85):
    '''
    Calculates PageRank for an undirected graph
    Returns: A list of tuples representing sentences and respective
    scores in descending order
    '''
    graph = obj.graph
    try:
        if initial_value == None: initial_value = 1.0 / len(graph.nodes())
    except:
        return obj.data
    scores = dict.fromkeys(graph.nodes(), initial_value)

    iteration_quantity = 0
    for iteration_number in xrange(100):
        iteration_quantity += 1
        convergence_achieved = 0
        for i in graph.nodes():
            rank = 1 - damping
            for j in graph.neighbors(i):
                neighbors_sum = sum([graph.get_edge_data(j, k)['weight'] for k in graph.neighbors(j)])
                rank += damping * scores[j] * graph.get_edge_data(j, i)['weight'] / neighbors_sum

            if abs(scores[i] - rank) <= CONVERGENCE_THRESHOLD:
                convergence_achieved += 1

            scores[i] = rank

        if convergence_achieved == len(graph.nodes()):
            break
    return sorted(scores.items(), key=itemgetter(1), reverse=True)

