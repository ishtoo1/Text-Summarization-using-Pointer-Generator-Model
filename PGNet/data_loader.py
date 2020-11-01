import torch
import random


class DataLoader(object):
        def __init__(self,data_doc,data_summ,vocab,
                     batch_size,max_len,max_summary_len):
                self.data=list(zip(data_doc,data_summ))
                self.batch_size=batch_size
                self.vocab=vocab
                self.max_len=max_len
                self.max_summary_len=max_summary_len
        
        def _get_data(self):
                data=random.choices(self.data,k=self.batch_size)
                
                return data
            
        def get_batch(self):
                data=self._get_data()
                doc,summ=zip(*data)
                doc,summ=list(doc),list(summ)
                
                max_len,max_len_summ=0,0
                for article in doc:
                        max_len=max(max_len,len(article))
                for summm in summ:
                        max_len_summ=max(max_len_summ,len(summm))
                max_len=min(max_len,self.max_len)
                max_len_summ=min(max_len_summ,self.max_summary_len)
                        
                data_x,data_y,doc_len,summ_len=[],[],[],[]
                for article in doc:
                        temp_x=[]
                        temp_x.append(self.vocab. \
                                      word_to_int[self.vocab.start])
                        for i,word in enumerate(article):
                                if word in self.vocab.word_to_int:
                                        temp_x. \
                                        append(self.vocab. \
                                               word_to_int[word])
                                else:
                                        temp_x. \
                                        append(self.vocab. \
                                               word_to_int[self.vocab.unk])
                                if i==max_len-1:
                                        break
                                
                        temp_x.append(self.vocab. \
                                      word_to_int[self.vocab.end])
                        lt=len(temp_x)-2
                        doc_len.append(lt+2)
                        for _ in range(max(0,max_len-lt)):
                                temp_x.append(self.vocab. \
                                      word_to_int[self.vocab.pad])
                                
                        data_x.append(temp_x)
                        
                for summary in summ:
                        temp_y=[]
                        temp_y.append(self.vocab. \
                                      word_to_int[self.vocab.start])
                        for i,word in enumerate(summary):
                                if word in self.vocab.word_to_int:
                                        temp_y. \
                                        append(self.vocab. \
                                               word_to_int[word])
                                else:
                                        temp_y. \
                                        append(self.vocab. \
                                               word_to_int[self.vocab.unk])
                                if i==max_len_summ-1:
                                        break
                                    
                        temp_y.append(self.vocab. \
                                      word_to_int[self.vocab.end])
                        lt=len(temp_y)-2
                        summ_len.append(lt+2)
                        for _ in range(max(0,max_len_summ-lt)):
                                temp_y.append(self.vocab. \
                                      word_to_int[self.vocab.pad])
                                
                        data_y.append(temp_y)
                        
                doc_data=torch.zeros(self.batch_size,max_len+2,dtype=torch.long).cuda()
                summ_data=torch.zeros(self.batch_size,max_len_summ+2,dtype=torch.long).cuda()
                doc_mask=torch.zeros(self.batch_size,max_len+2,dtype=torch.long).cuda()
                summ_mask=torch.zeros(self.batch_size,max_len_summ+2,dtype=torch.long).cuda()
                
                for i in range(self.batch_size):
                        doc_data[i]=torch.tensor(data_x[i])
                        summ_data[i]=torch.tensor(data_y[i])
                        summ_mask[i][:summ_len[i]]=1
                        doc_mask[i][:doc_len[i]]=1
                        
                return doc_data,doc_mask,summ_data,summ_mask
            
        
class Test_loader(DataLoader):
        def __init__(self,data_doc,data_summ,vocab,
                         max_len,max_summary_len):
                self.data=list(zip(data_doc,data_summ))
                self.len=len(self.data)
                self.batch_size=1
                self.vocab=vocab
                self.max_len=max_len
                self.max_summary_len=max_summary_len
                self.counter=0
                
        def _get_data(self):
                data=self.data[self.counter:self.counter+self.batch_size]
                self.counter+=self.batch_size
                if self.counter==len(self.data):
                        self.counter=0
                        
                return data