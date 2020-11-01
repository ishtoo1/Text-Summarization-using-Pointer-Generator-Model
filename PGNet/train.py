import torch
import torch.optim as optim

import argparse
import os

from utils import Vocab,read_from_file,read_vocab
from data_loader import Test_loader,DataLoader
from model import Encoder,Decoder,Embeddings
from decode import BeamSearch

def train(embeddings,encoder,decoder,generator,trainloader,valloader,
          iters,lambda_,lr,max_grad_norm,initial_accum_val,threshold):
    
        params=list(embeddings.parameters())+ \
        list(encoder.parameters())+list(decoder.parameters())
        optimizer=optim.Adagrad(params,lr=lr,initial_accumulator_value=initial_accum_val)
        
        scalar,in_loss=0,10000000
        for i in range(iters):
                encoder.train()
                embeddings.train()
                decoder.train()
                
                optimizer.zero_grad()
                if i>=threshold:
                        is_coverage=True
                else:
                        is_coverage=False
                
                loss=train_one_batch(embeddings,encoder,decoder,
                                     trainloader,is_coverage,lambda_)
                scalar+=loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(),max_grad_norm)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(),max_grad_norm)
                torch.nn.utils.clip_grad_norm_(embeddings.parameters(),max_grad_norm)
                
                optimizer.step()
                
                if (i+1)%100==0:
                        statement='training iterations= {}, training loss= {}'. \
                        format(i+1,scalar/100)
                        t_t=open('feed.txt','a')
                        t_t.write(statement+'\n')
                        t_t.close()
                        scalar=0
                        
                if (i+1)%5000==0:
                        encoder.eval()
                        embeddings.eval()
                        decoder.eval()
                        loss_,r1,r2,rl,gen,ref=evaluate(embeddings,encoder,decoder,valloader,
                                 is_coverage,generator,lambda_)
                        loss__=sum(loss_)/len(loss_)
                        r1__=sum(r1)/len(loss_)
                        r2__=sum(r2)/len(loss_)
                        rl__=sum(rl)/len(loss_)
                        
                        if loss__<in_loss:
                                torch.save({
                                        'embed_params':embeddings.state_dict(),
                                        'encoder_params':encoder.state_dict(),
                                        'decoder_params':decoder.state_dict()
                                        },'model.pth')
                                in_loss=loss__
                                
                        for i in range(len(loss_)):
                                statement='validation loss= {}, ROUGE-1= {},\
                                ROUGE-2= {}, ROUGE-L= {}, generated summary= {}\
                                reference summary= {}' \
                                .format(loss_[i],r1[i],r2[i],rl[i],gen[i],ref[i])
                                t_v=open('val_feed.txt','a')
                                t_v.write(statement+'\n')
                                t_v.close()
                                
                        statement='average loss= {}, average ROUGE-1= {},\
                        average ROUGE-2= {}, average ROUGE-L= {}' \
                        .format(loss__,r1__,r2__,rl__)
                        t_v=open('val_feed.txt','a')
                        t_v.write(statement+'\n\n\n')
                        t_v.close()
                        
def train_one_batch(embeddings,encoder,decoder,trainloader,
                    is_coverage,lambda_):
        doc,doc_mask,summ,summ_mask=trainloader.get_batch()
        max_steps=summ.shape[-1]
        
        enc_embeds=embeddings(doc)
        enc_out,(h,c)=encoder(enc_embeds,doc_mask)
        dec_embeds=embeddings(summ)
        
        loss=0
        if is_coverage:
                coverage=torch.zeros((enc_embeds.shape[0], \
                                      1,enc_embeds.shape[1]),requires_grad=True).cuda()
        else:
                coverage=None
        for i in range(max_steps-1):
                dec_inp=dec_embeds[:,i,:].view(trainloader.batch_size,1,-1)
                pw,attn,coverage,h,c= \
                decoder(dec_inp,enc_out,doc_mask,h,c,doc,is_coverage,coverage)
                
                pw=pw.squeeze(1)
                gold=summ[:,i+1].view(-1,1)
                probs=pw.gather(-1,gold).squeeze(-1)
                log_probs=-torch.log(probs)
                
                if is_coverage:
                        c_temp=coverage-attn
                        coverage_loss=torch.sum(torch.min(c_temp,attn),dim=-1)
                        coverage_loss=coverage_loss*lambda_
                        coverage_loss=coverage_loss.squeeze(-1)
                        log_probs+=coverage_loss
                        
                log_probs=log_probs*summ_mask[:,i+1]
                loss+=log_probs.sum()
                
        return loss/trainloader.batch_size
                        
def evaluate(embeddings,encoder,decoder,dataloader,
             is_coverage,generator,lambda_):
        
        r1_,r2_,rl_,scalar,gen_,ref_=[],[],[],[],[],[]
        for _ in range(dataloader.len):
                r1,r2,rl,gen,ref=generator.beam_search(embeddings,encoder,
                                       decoder,dataloader,is_coverage,True)
                r1_.append(r1)
                r2_.append(r2)
                rl_.append(rl)
                gen_.append(gen)
                ref_.append(ref)
                
        for _ in range(dataloader.len):
                loss=train_one_batch(embeddings,encoder,decoder,
                                     dataloader,is_coverage,lambda_)
                scalar.append(loss.item())
        
        return scalar,r1_,r2_,rl_,gen_,ref_

def test(embeddings,encoder,decoder,dataloader,
            generator,lambda_):
        
        r1_,r2_,rl_,gen_,ref_=[],[],[],[],[]
        for _ in range(dataloader.len):
                r1,r2,rl,gen,ref=generator.beam_search(embeddings,encoder,decoder,
                                               dataloader,True,True)
                r1_.append(r1)
                r2_.append(r2)
                rl_.append(rl)
                gen_.append(gen)
                ref_.append(ref)
                
        r1_avg,r2_avg,rl_avg=sum(r1_)/dataloader.len, \
        sum(r2_)/dataloader.len,sum(rl_)/dataloader.len
    
        for i in range(len(r1)):
                statement='For {}th testing example, ROUGE-1= {}, ROUGE-2= {},\
                ROUGE-L={}, generated summary= {}\
                reference summary={}'.format(i+1,r1_[i],r2_[i],rl_[i],gen_[i],ref_[i])
                
                f_t=open('test_feed.txt','a')
                f_t.write(statement+'\n\n\n')
                f_t.close()
        statement='Average ROUGE-1= {}, Average ROUGE-2= {}, Average ROUGE-L={}'. \
        format(r1_avg,r2_avg,rl_avg)
        f_t=open('test_feed.txt','a')
        f_t.write(statement+'\n')
        f_t.close()

def main(args):
        train_doc,train_summ,test_doc, \
        test_summ,val_doc,val_summ=read_from_file(args.data_file)
        vocab=read_vocab(args.path_to_vocab)
        
        embeddings=Embeddings(args.embed_size,args.vocab_size).cuda()
        encoder=Encoder(args.embed_size,args.hidden_size).cuda()
        decoder=Decoder(args.embed_size,args.hidden_size,args.vocab_size).cuda()
        generator=BeamSearch(vocab,args.max_decode_len,
                             args.min_decode_len,args.beam_size)
        
        trainloader=DataLoader(train_doc,train_summ,vocab,
                               args.batch_size,args.max_doc_len,args.max_summ_len)
        testloader=Test_loader(test_doc,test_summ,vocab,
                               args.max_doc_len,args.max_summ_test_len)
        valloader=Test_loader(val_doc,val_summ,vocab,
                               args.max_doc_len,args.max_summ_test_len)
        
        if args.use_pretrained:
            
                params=torch.load(args.pretrained_model)
                embeddings.load_state_dict(params['embed_params'])
                encoder.load_state_dict(params['encoder_params'])
                decoder.load_state_dict(params['decoder_params'])
                
                test(embeddings,encoder,decoder,testloader,
                     generator,args.lambda_)

        train(embeddings,encoder,decoder,generator,
              trainloader,valloader,args.iterations,args.lambda_,
              args.lr,args.max_grad_norm,args.initial_accum_val,args.threshold)
        
        test(embeddings,encoder,decoder,testloader,
             generator,args.lambda_)

def setup():
        parser=argparse.ArgumentParser()
        
        parser.add_argument('--embed_size',type=int,default=128)
        parser.add_argument('--hidden_size',type=int,default=256)
        parser.add_argument('--vocab_size',type=int,default=50000)
        parser.add_argument('--lr',type=float,default=0.15)
        parser.add_argument('--lambda_',type=float,default=1)
        parser.add_argument('--iterations',type=int,default=233000)
        parser.add_argument('--epochs',type=int,default=120)
        parser.add_argument('--initial_accum_val',type=float,default=0.1)
        parser.add_argument('--beam_size',type=int,default=5)
        parser.add_argument('--max_decode_len',type=int,default=120)
        parser.add_argument('--min_decode_len',type=int,default=35)
        parser.add_argument('--batch_size',type=int,default=16)
        parser.add_argument('--max_grad_norm',type=float,default=2.0)
        parser.add_argument('--data_file',type=str,default='/home/samyak/train.pickle')
        parser.add_argument('--max_summ_len',type=int,default=120)
        parser.add_argument('--max_doc_len',type=int,default=400)
        parser.add_argument('--max_summ_test_len',type=int,default=120)
        parser.add_argument('--use_pretrained',type=bool,default=False)
        parser.add_argument('--pretrained_model', type=str,default=os.getcwd()+'/pgtmodel.pth')
        parser.add_argument('--path_to_vocab',type=str,default='/home/samyak/vocab.pickle')
        parser.add_argument('--threshold',type=int,default=230000)
        
        args=parser.parse_args()
        
        return args
    
if __name__=='__main__':
        args=setup()
        main(args)