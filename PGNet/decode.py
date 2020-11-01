import torch
from rouge import Rouge


class Beamobj(object):
        def __init__(self,tokens,log_prob,h,c,coverage):
                self.tokens=tokens
                self.log_prob=log_prob
                self.h=h
                self.c=c
                self.coverage=coverage
                
        def get_token(self):
                return self.tokens[-1]
            
        def get_prob(self):
                return self.log_prob/len(self.tokens)
            
        def get_child(self,tokens,prob,h,c,coverage):
                return Beamobj(tokens,
                               prob,
                               h,c,coverage)
                
                
class BeamSearch(object):
        def __init__(self,vocab,max_decoding_len,min_len,beam_size):
                self.vocab=vocab
                self.max_decode_len=max_decoding_len
                self.beam_size=beam_size
                self.min_len=min_len
                self.rouge=Rouge()

        def beam_search(self,embeddings,encoder,decoder,dataloader,
                        is_coverage,return_summ=True):
                doc,doc_mask,summ,summ_mask=dataloader.get_batch()
                doc_=embeddings(doc)
                enc_out,(h,c)=encoder(doc_,doc_mask)
                
                beam,results=[],[]
                if is_coverage:
                        coverage=torch.zeros(1,1,enc_out.shape[-2]).cuda()
                else:
                        coverage=None
                beam.append(Beamobj([self.vocab.word_to_int[self.vocab.start]],
                                    0,h,c,coverage))
                
                iters=0
                while iters<self.max_decode_len and len(results)<self.beam_size:
                        inps,hs,hc,coverages=[],[],[],[]
                        for instance in beam:
                                inps.append(instance.get_token())
                                hs.append(instance.h)
                                hc.append(instance.c)
                                coverages.append(instance.coverage)
                                
                        inps=torch.tensor(inps).view(-1,1)
                        hc=torch.cat(hc,dim=1)
                        hs=torch.cat(hs,dim=1)
                        
                        if is_coverage:
                                coverages=torch.cat(coverages,dim=0)
                        else:
                                coverages=None
                        
                        dec_embeds=embeddings(inps)
                        doc_=doc.expand(dec_embeds.shape[0],-1)
                        enc_out_=enc_out.expand(dec_embeds.shape[0],-1,-1)
                        pw,attn,coverage,h,c=decoder(dec_embeds,enc_out_,
                                                     doc_mask,hs,hc,doc_,is_coverage,coverages)
                        pw=pw.squeeze(1)
                        probs,indi=torch.topk(pw,self.beam_size,dim=-1)
                        
                        news=[]
                        for i,instance in enumerate(beam):
                                tokens=instance.tokens
                                pb=instance.log_prob
                                h_s=h[0][i].view(1,1,-1)
                                h_c=c[0][i].view(1,1,-1)
                                if is_coverage:
                                        cov=coverage[i].view(1,1,-1)
                                else:
                                        cov=None
                                
                                for j in range(self.beam_size):
                                        if is_coverage:
                                                cov_=cov+attn[i].view(1,1,-1)
                                        else:
                                                cov_=cov
                                                
                                        news.append(instance. \
                                                    get_child(tokens+[indi[i][j].item()],
                                                              pb-torch.log(probs[i][j]).item(),
                                                              h_s,h_c,cov_))
                                                    
                        beam=[]
                        news=sorted(news,key=lambda x:x.get_prob(),reverse=True)
                        for i,instance in enumerate(news):
                                if instance.get_token()==self. \
                                vocab.word_to_int[self.vocab.end]:
                                        if len(instance.tokens)>self.min_len:
                                                results.append(instance)
                                else:
                                        beam.append(instance)
                                if len(beam)==self.beam_size or  \
                                len(results)==self.beam_size:
                                        break
                                    
                        iters+=1
                        
                if len(results)==0:
                        results=beam
                results=sorted(results,key=lambda x:x.get_prob(),reverse=True)
                
                if not return_summ:
                        rouge_1,rouge_2,rouge_l=self._get_rouge(results[0],summ,return_summ)
                        return rouge_1,rouge_2,rouge_l
                else:
                        rouge_1,rouge_2,rouge_l,gen,ref=self. \
                        _get_rouge(results[0],summ,return_summ)
                        return rouge_1,rouge_2,rouge_l,gen,ref

        def _get_rouge(self,result,summ,return_summ):
                reference=summ.squeeze()
                ref,gen='',''
                
                for i in range(1,reference.shape[0]-1):
                        ref+=self.vocab.int_to_word[int(reference[i])]
                        ref+=' '
                for i in range(1,len(result.tokens)):
                        if i!=len(result.tokens)-1:
                                gen+=self.vocab.int_to_word[result.tokens[i]]
                                gen+=' '
                        elif i==len(result.tokens) and  \
                        result.tokens[i]!=self.vocab.word_to_int[self.vocab.end]:
                                gen+=self.vocab.int_to_word[result.tokens[i]]
                                gen+=' '
                                
                if ref[-1]==' ':
                        ref=ref[:-1]
                if gen[-1]==' ':
                        gen=gen[:-1]
                
                rg=self.rouge.get_scores(gen,ref)
                r1,r2,rl=rg[0]["rouge-1"]['f'], \
                rg[0]["rouge-2"]['f'],rg[0]["rouge-l"]['f']
                
                if not return_summ:
                        return r1,r2,rl
                else:
                        return r1,r2,rl,gen,ref
