import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class Embeddings(nn.Module):
        def __init__(self,embed_size,vocab_size):
                super(Embeddings,self).__init__()
                self.embeddings=nn.Embedding(vocab_size,embed_size,padding_idx=0)
                
        def forward(self,input):
                output=self.embeddings(input)
                return output


class Encoder(nn.Module):
        def __init__(self,input_size,hidden_size):
                super(Encoder,self).__init__()
                self.lstm=nn.LSTM(input_size,hidden_size,bidirectional=True,batch_first=True)
                self.wh=nn.Linear(2*hidden_size,2*hidden_size)
                
                self.reducer=Reduce_state(hidden_size)
                
        def forward(self,input,input_mask):
                input_mask=torch.sum(input_mask,dim=-1)  
                packed_seq=pack_padded_sequence \
                (input,input_mask,batch_first=True,enforce_sorted=False)
                output,(hidden,cell_state)=self.lstm(packed_seq)
                output,_=pad_packed_sequence(output,batch_first=True)
                output=self.wh(output)
                
                hidden,cell_state=self.reducer(hidden,cell_state)
                
                return output,(hidden,cell_state)


class Reduce_state(nn.Module):
        def __init__(self,hidden_size):
                super(Reduce_state,self).__init__()
                self.h_project=nn.Linear(2*hidden_size,hidden_size)
                self.c_project=nn.Linear(2*hidden_size,hidden_size)
            
        def forward(self,h,c):
                h=torch.cat([h[0],h[1]],dim=-1)
                c=torch.cat([c[0],c[1]],dim=-1)
                
                h=F.relu_(self.h_project(h))
                c=F.relu_(self.c_project(c))
                h=h.unsqueeze(0)
                c=c.unsqueeze(0)
                
                return h,c
                
            
class Decoder(nn.Module):
        def __init__(self,embed_size,hidden_size,vocab_size):
                super(Decoder,self).__init__()
                self.lstm=nn.LSTM(embed_size,hidden_size,batch_first=True)
                self.v=nn.Linear(2*hidden_size,1)
                self.wc=nn.Linear(1,2*hidden_size)
                self.ws=nn.Linear(2*hidden_size,2*hidden_size)
                
                self.out1=nn.Linear(3*hidden_size,hidden_size)
                self.out2=nn.Linear(hidden_size,vocab_size+4)
                
                self.p_gen=nn.Linear(2*2*hidden_size+embed_size,1)
                
        def forward(self,dec_inp,enc_out,enc_mask,h,c,
                    enc_ind,is_coverage,coverage=None):
                dec_out,(h,c)=self.lstm(dec_inp,(h,c))
                s=torch.cat([h,c],dim=-1).squeeze(0).unsqueeze(1)
                context_vector,attn,coverage=self. \
                calculate_attention(enc_out,s,coverage,enc_mask,is_coverage)
                
                pw=self.out2(self. \
                             out1(torch.cat([context_vector,h. \
                                             squeeze(0).unsqueeze(1)],dim=-1)))
                pw=F.softmax(pw,dim=-1)
                pgen=self.calculate_pgen(context_vector,s,dec_inp)
                pw=pw*pgen
                atd=attn*(1-pgen)
                
                for i in range(atd.shape[0]):
                        for j in range(atd.shape[-1]):
                                pw[i][0][enc_ind[i][j]]+=atd[i][0][j]
                                
                return pw,attn,coverage,h,c
    
        def calculate_attention(self,enc_out,s,coverage,enc_mask,is_coverage):
                if is_coverage:
                        wc_=self.wc(coverage.squeeze(1).unsqueeze(-1))
                else:
                        wc_=0
                        
                ws_=self.ws(s)
                summation=torch.tanh(enc_out+ws_+wc_)
                e=self.v(summation).squeeze(-1)
                enc_mask_=(1-enc_mask)*(-1e9)
                e+=enc_mask_
                attn=F.softmax(e,dim=-1)
                attn=attn.unsqueeze(1)
                
                context_vector=torch.bmm(attn,enc_out)
                if is_coverage:
                        coverage_new=coverage+attn
                        return context_vector,attn,coverage_new
                
                return context_vector,attn,coverage
            
        def calculate_pgen(self,context_vector,s,dec_inp):
                vec=torch.cat([context_vector,s,dec_inp],dim=-1)
                pgen=torch.sigmoid(self.p_gen(vec))
                
                return pgen