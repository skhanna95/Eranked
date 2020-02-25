from tqdm import tqdm
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def makeVocabDict(pd_df,col_list):
    
    word_to_ix = {}
    word_to_ix['<UNK>'] = 0
    word_to_ix['<PAD>'] = 1
    word_to_ix['<EOS>'] = 2
    
    
    for i in tqdm(range(len(pd_df)),desc='makeVocabDict'):
        for cl in col_list:
            for word in str(pd_df[cl][i]).split():
                if word.lower() not in word_to_ix:
                    word_to_ix[word.lower()] = len(word_to_ix)
            
    return word_to_ix

def getRefDict(filename):
    refer_dict = {}
    with open('data/reference/'+filename) as lines:
        
        for line in lines:
            split_line = line.split('||')
            refer_dict[split_line[0]] = int(split_line[1].replace('\n',''))

    return refer_dict

def makeDataDict(pd_df, id_col = 'product_uid',title_col = 'product_title',
               query_col='search_term',brand_col='brand',description='product_description',
               relevance='relevance',train=True,title='train'):
    
    
    if train:
        train_data_dict = {}

        for i in tqdm(range(len(pd_df)),desc='transformingData-'+title):
            if train_data_dict.get(pd_df[query_col][i]):
                train_data_dict[pd_df[query_col][i]]['product_uid'].append(pd_df[id_col][i])
                train_data_dict[pd_df[query_col][i]]['titles'].append(pd_df[title_col][i])
                train_data_dict[pd_df[query_col][i]]['desc'].append(pd_df[description][i])
                train_data_dict[pd_df[query_col][i]]['brand'].append(pd_df[brand_col][i])
                train_data_dict[pd_df[query_col][i]]['relevance'].append(pd_df[relevance][i])
            else:
                train_data_dict[pd_df[query_col][i]]={}
                train_data_dict[pd_df[query_col][i]]['product_uid'] =[ pd_df[id_col][i] ]
                train_data_dict[pd_df[query_col][i]]['titles'] =[ pd_df[title_col][i] ]
                train_data_dict[pd_df[query_col][i]]['desc'] =[ pd_df[description][i] ]
                train_data_dict[pd_df[query_col][i]]['brand'] =[ pd_df[brand_col][i] ]
                train_data_dict[pd_df[query_col][i]]['relevance'] =[ pd_df[relevance][i] ]
                
                
        return train_data_dict

def readFiles(file_name):
    data_dict = {}
    with open(file_name) as lines:
        for line in lines:
            line_split = line.split()
            data_dict[line_split[0]] = {}
            for child in line_split[1].split(','):
                csplit = child.split('-')
                data_dict[line_split[0]][csplit[0]] = csplit[1]
                
    return data_dict

def makeIDXbasedTensors(seq,word_to_ix,unk):
    idxs = []
    
    if unk not in word_to_ix:
        idxs = [word_to_ix[w] for w in seq]
    else:
        idxs = [word_to_ix[w] for w in map(lambda w: unk if w not in word_to_ix else w, seq)]
        
    return torch.tensor(idxs, dtype=torch.long).to(device)

def makeTensorData(pd_df,word_to_ix,title):
    
    data_dict = makeDataDict(pd_df,title=title)

    master_list = []

    

    if "test" in title.lower():
        ref_dict = getRefDict('s_dict_test.txt')
    else:
        ref_dict = getRefDict('s_dict_train.txt')
    
    for key,value in tqdm(data_dict.items(),desc='makeTensorData-'+title):
        
        titles_tensors = []
        rel_tensors = []
        product_uid = []
        desc = []

        for TT in value['titles']:
            titles_tensors.append(makeIDXbasedTensors(TT,word_to_ix,'<UNK>'))
            
        for RR in value['relevance']:
            rel_tensors.append(torch.tensor(RR, dtype=torch.float).to(device))

        for pid in value['product_uid']:
            product_uid.append(torch.tensor(pid, dtype=torch.long).to(device))

        for DD in value['desc']:
            desc.append(makeIDXbasedTensors(DD,word_to_ix,'<UNK>'))
        
        assert len(desc) == len(titles_tensors)

        for i in range(len(titles_tensors)):
            titles_tensors[i]= torch.cat(tensors=[titles_tensors[i],desc[i]])


        max_TT = 0
        for mt in titles_tensors:
            if len(mt) > max_TT:
                max_TT = len(mt)
                
        for i in range(len(titles_tensors)):
            crLen = max_TT - len(titles_tensors[i])
            if crLen > 0:
                padder = torch.tensor([1]*crLen, dtype=torch.long).to(device)
                titles_tensors[i] = torch.cat([titles_tensors[i],padder])
            
        master_list.append(
            (   torch.stack(tensors=product_uid).to(device),
                makeIDXbasedTensors(key,word_to_ix,'<UNK>'),
                torch.stack(tensors=titles_tensors).to(device),
                torch.stack(tensors=rel_tensors).to(device),
                torch.tensor([ref_dict[key]])
            )
        )
    
    return master_list

def makeOutFile(op_dict,op_file):
    if not os.path.exists('output'):
        os.mkdir('output')
        
    with open('output/'+op_file,'w') as ft:
        for k,v in op_dict.items():
            app_lst = []

            for sk,sv in v.items():
                app_lst.append( sk + '-' + sv )

            ft.write( k + ' ' + ",".join(app_lst) + '\n')