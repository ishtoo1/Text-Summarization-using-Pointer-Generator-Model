import tensorflow_datasets as tfds
import nltk
from collections import Counter
import pickle

def load_data(split):
        ds=tfds.load('cnn_dailymail', split=split, shuffle_files=True) 
        documents,summary=[],[]
        
        for examples in ds:
                summ=examples['highlights'].numpy(). \
                decode('utf-8').replace('\n',' '). \
                replace("'","' ").lower()
                doc=examples['article'].numpy(). \
                decode('utf-8').replace('\n',' '). \
                replace("'","' ").lower()
                
                doc=nltk.word_tokenize(doc)
                summ=nltk.word_tokenize(summ)
                documents.append(doc)
                summary.append(summ)
                
        return documents,summary

                
class Vocab(object):
        def __init__(self,vocab_size,doc,summ):
                self.doc=doc
                self.summ=summ
                self.pad='<pad>'
                self.unk='<unk>'
                self.start='<s>'
                self.end='</s>'
                self.vocab=set()
                self.word_to_int={}
                self.int_to_word={}
                
                self.vocab_size=vocab_size
                self.get_vocab()
                
        def get_vocab(self):
                word_count=Counter()
                for article in self.doc:
                        for word in article:
                                word_count[word]+=1
                for summa in self.summ:
                        for word in summa:
                                word_count[word]+=1
                                
                vocab_list=[]
                for word in word_count:
                        vocab_list.append((word_count[word],word))
                vocab_list.sort()
                vocab_list=vocab_list[-self.vocab_size:]
                for tup in vocab_list:
                        self.vocab.add(tup[1])
                        
                self.get_word_indexing()
                        
        def get_word_indexing(self):
                ind=4
                for word in self.vocab:
                        self.word_to_int[word]=ind
                        self.int_to_word[ind]=word
                        ind+=1
                        
                self.word_to_int[self.pad],self.word_to_int[self.start]=0,1
                self.word_to_int[self.end],self.word_to_int[self.unk]=2,3
                
                self.int_to_word[0],self.int_to_word[1]=self.pad,self.start
                self.int_to_word[2],self.int_to_word[3]=self.end,self.unk
                
def store_in_file(file_path):
        train_doc,train_summ=load_data('train')
        test_doc,test_summ=load_data('test')
        val_doc,val_summ=load_data('validation')
        store={}
        
        store['train_doc'],store['train_summ']=train_doc,train_summ
        store['test_doc'],store['test_summ']=test_doc,test_summ
        store['val_doc'],store['val_summ']=val_doc,val_summ
        
        file=open(file_path+'.pickle','wb')
        pickle.dump(store,file)
        
def read_from_file(file_name):
        file=open(file_name,"rb")
        store=pickle.load(file)
        
        return store['train_doc'],store['train_summ'], \
        store['test_doc'],store['test_summ'], \
        store['val_doc'],store['val_summ']
        
def store_vocab(file_name,vocab):
        file=open(file_name,'wb')
        store={}
        store['vocab']=vocab
        pickle.dump(store,file)

def read_vocab(file_name):
        file=open(file_name,'rb')
        store=pickle.load(file)
        
        return store['vocab']