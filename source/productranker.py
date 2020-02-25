import os
import time
import copy
import torch
import operator
import optparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from cpsir import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetricCalc(object):

    def __init__(self):
        self.average = 0
        self.sum = 0
        self.count = 0

    def maintain(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count

class ProductRanker(object):
    
    def __init__(self,src_dict, unk="<UNK>", embedding_dim=128, hidden_dim=64,
                 batch_size = 1,learning_rate = 0.0001, weight_decay=0.95,
                 state_dict=False,optimizer=None):
        self.src_dict = src_dict
        self.unk = unk
        self.word_to_ix = src_dict

        self.network = CPSIR(
                        embedding_dim = embedding_dim,
                        hidden_dim = hidden_dim,
                        vocab_size = len(self.word_to_ix),
                        tagset_size = 128,
                        batch_size = batch_size
        )
        self.criterion = torch.nn.MSELoss()
        self.network.train()

        params = [p for p in self.network.parameters() if p.requires_grad] 

        self.optimizer = optim.SGD(params, learning_rate,
                                   weight_decay=weight_decay)

        if state_dict:
            self.network.load_state_dict(state_dict)
                        

    def forward(self, rows, clip_grad = 1e-8):
        
        search_queries = rows[1]
        products = rows[2]
        relevances = rows[3]

        scores = self.network(search_queries, products)
        
        loss = self.criterion(-scores, relevances.unsqueeze(0))
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),clip_grad)#1e-12
        
        self.optimizer.step()
        
        return loss

    def predict(self, rows):
        self.network.eval()
        
        queries = rows[1]
        documents = rows[2]
        
        scores = self.network(queries, documents)
        
        scores = f.softmax(scores, dim=-1)
        
        return scores

    def savemodel(self, filename):
        
        state_dict = copy.copy(self.network.state_dict())
        
        params = {
          'state_dict': state_dict,
          'src_dict': self.src_dict,
        }
        
        try:
            torch.save(params, filename)
            
        except:
            print('Error Saving the model')

    def savecheckpoint(self, filename, epoch):
        
        params = {
          'state_dict': self.network.state_dict(),
          'src_dict': self.src_dict,
          'epoch': epoch,
          'optimizer': self.optimizer.state_dict(),
          }
        
        try:
            torch.save(params, filename)
            
        except:
            print('Error Saving the model')
          
    @staticmethod
    def loadmodel(filename):
        params = torch.load(filename, map_location=lambda stg, loc: stg)
        return ProductRanker( src_dict = params['src_dict'], state_dict = params['state_dict'])
        
    @staticmethod
    def loadcheckpoint(filename):
        params = torch.load(filename, map_location='cpu')
        
        epoch = params['epoch']
        
        model = ProductRanker(src_dict = params['src_dict'],state_dict = params['state_dict'],optimizer=params['optimizer'])
        
        return model, epoch

def precision(predictions,p_uid,sid, result_ref):
    predictions = predictions.squeeze(0)
    matches = 0
    
    for pi in range(len(p_uid)):
        
        if predictions[pi].item() == int(result_ref[str(sid.item())][str(p_uid[pi].item())]):
            matches += 1

    return matches/len(predictions)


def validate_result(data_loader, model, batch_size,result_ref):
    
    metric = MetricCalc()
    op_dict = {}
    
    with torch.no_grad():
      
        for rows in tqdm(data_loader):
            p_uid,_,_,relevance,sid = rows
            op_dict[str(sid.item())] = {}
            
            scores = model.predict(rows)
            # set_trace()
            
            predictions = torch.argsort(torch.argsort(scores,descending=True))
            sq_predictions = predictions.squeeze(0)
            
            for i in range(len(p_uid)):
                 op_dict[str(sid.item())][str(p_uid[i].item())] = str(sq_predictions[i].item())
                    
            metric.maintain(precision(predictions,p_uid,sid, result_ref))
    
    return metric.average,op_dict

def train(learning_rate,epochs,embedding_dim,hidden_dim,
          weight_decay,lr_decay,modelfile,unktoken,batch_size=1):

    print('Loading Train Data...')
    full_data = pd.read_csv('data/train_dssm3.csv')
    train_data = pd.read_csv('data/train_final.csv')

    src_dict = makeVocabDict(full_data,['product_title','search_term','brand','product_description'])

    trainDataset = makeTensorData(train_data,src_dict,title='train')

    print('Loading Dev Data...')
    dev_data = pd.read_csv('data/dev_final.csv')
    devDataset = makeTensorData(dev_data,src_dict,title='dev')
    
    model = ProductRanker(src_dict = src_dict,
                          unk=unktoken,
                          embedding_dim = embedding_dim,
                          hidden_dim = hidden_dim,
                          learning_rate = learning_rate,
                          weight_decay=weight_decay
                          )
 
    best_valid = 0
    
    result_ref = readFiles('data/reference/dev.out')
    
    for i in range(epochs):
        
        metric = MetricCalc()
    
        model.optimizer.param_groups[0]['lr'] =  model.optimizer.param_groups[0]['lr'] * lr_decay

        pbar = tqdm(enumerate(trainDataset))

        for idx,rows in pbar:
            loss = model.forward(rows)
            metric.maintain(loss.item())

            pbar.set_description("Epoch = {} Loss = {}".format(i+1,round(metric.average /10, 5)))

            torch.cuda.empty_cache()

        result,op_dict = validate_result(devDataset, model, batch_size,result_ref )
        print(result)
        
        if result > best_valid:
            model.savemodel(modelfile)
            best_valid = result
            makeOutFile(op_dict,'dev.out')

        model.savecheckpoint(modelfile + '.checkpoint', i + 1)

def test(modelfile,batch_size=1):
    
    model = ProductRanker.loadmodel(modelfile)
    test_data = pd.read_csv('data/test_final.csv').dropna().reset_index().drop(['index'],axis=1)
    trainDataset = makeTensorData(test_data,model.src_dict,title='test')

    result_ref = readFiles('data/reference/test.out')
    
    result,op_dict = validate_result(trainDataset, model, batch_size,result_ref )
    print(result)
   
    makeOutFile(op_dict,'test.out')
    
if __name__ == '__main__':

    optparser = optparse.OptionParser()
    optparser.add_option("-l", "--learning_rate", dest='learning_rate', default=0.0001, help="Learning Rate")
    optparser.add_option("-i", "--epochs", dest="epochs", default=10, help="Epochs")
    optparser.add_option("-e", "--embedding_dim", dest="embedding_dim", default=128, help="Embedding Dimension")
    optparser.add_option("-d", "--hidden_dim", dest='hidden_dim', default=64, help="Hidden Dimension")
    optparser.add_option("-w", "--weight_decay", dest="weight_decay", default=0.95, help="Weight Decay")
    optparser.add_option("-r", "--lr_decay", dest="lr_decay", default=0.95, help="Learning Decay")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default="products_ranker.pt", help="Model File Name")
    optparser.add_option("-u", "--unktoken", dest="unktoken", default="<UNK>", help="Unknown word Token")
    optparser.add_option("-t", "--testmode", dest="testmode", default=True, help="Run Only Test (true/false)")


    (opts, _) = optparser.parse_args()

    learning_rate = float(opts.learning_rate)
    epochs = int(opts.epochs)
    embedding_dim = int(opts.embedding_dim)
    hidden_dim = int(opts.hidden_dim)
    weight_decay = float(opts.weight_decay)
    lr_decay =  float(opts.lr_decay)
    modelfile = opts.modelfile
    unktoken = opts.unktoken
    testmode = opts.testmode
    

    if not isinstance(testmode, bool):
        if "true" in testmode.lower():
            testmode = True
        else:
            testmode = False

    if not os.path.exists(modelfile) and testmode:
        print('Model File not found:')
        testmode = False

    if testmode:
        print('Test mode on:')
        test(modelfile)

    else:
        print('Training mode on:')
        train(learning_rate,epochs,embedding_dim,hidden_dim,
          weight_decay,lr_decay,modelfile,unktoken)
        test(modelfile)