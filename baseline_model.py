
# imports


import numpy as np
import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost
from xgboost import plot_importance

def main():
    # using the training data

    train_df=pd.read_csv('data/train_new.csv')
    train_df.head(5)


    # finding word counts and substrings to make features

    def str_common_word(str1, str2):
        str1, str2 = str1.lower(), str2.lower()
        words, count = str1.split(), 0
        for word in words:
            if str2.find(word)>=0:
                count+=1
        return count
        
    def str_whole_word(str1, str2, i_):
        str1, str2 = str1.lower().strip(), str2.lower().strip()
        count = 0
        while i_ < len(str2):
            i_ = str2.find(str1, i_)
            if i_ == -1:
                return count
            else:
                count += 1
                i_ += len(str1)
        return count


    # making features using counts, term frequency of search in title and description etc.


    def features(df):
        df['word_len_of_search_term'] = df['search_term'].apply(lambda x:len(x.split())).astype(np.int64)
        df['word_len_of_title'] = df['product_title'].apply(lambda x:len(x.split())).astype(np.int64)
        df['word_len_of_description'] = df['product_description'].apply(lambda x:len(x.split())).astype(np.int64)
        df['word_len_of_brand'] = df['brand'].apply(lambda x:len(x.split())).astype(np.int64)
        
        # Create a new column that combine "search_term", "product_title" and "product_description"
        df['complete_product_desc'] = df['search_term']+"\t"+df['product_title'] +"\t"+df['product_description']
        
        # Number of times the entire search term appears in product title. 
        df['search_in_title'] = df['complete_product_desc'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
        
        # Number of times the entire search term appears in product description
        df['search_in_description'] = df['complete_product_desc'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
        
        # Number of words that appear in search term also appear in product title.
        df['word_in_title'] = df['complete_product_desc'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
        
        # Number of words that appear in search term also appear in production description.
        df['word_in_description'] = df['complete_product_desc'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
        
        # The ratio of product title word length to search term word length
        df['query_title_len_prop']=df['word_len_of_title']/df['word_len_of_search_term']
        
        # The ratio of product description word length to search term word length
        df['query_desc_len_prop']=df['word_len_of_description']/df['word_len_of_search_term']
        
        # The ratio of product title and search term common word count to search term word count
        df['ratio_title'] = df['word_in_title']/df['word_len_of_search_term']
        
        # The ratio of product description and search term common word count to search term word count.
        df['ratio_description'] = df['word_in_description']/df['word_len_of_search_term']
        
        # new column that combine "search_term", "brand" and "product_title".
        df['attr'] = df['search_term']+"\t"+df['brand']+"\t"+df['product_title']
        
        # Number of words that appear in search term also apprears in brand.
        df['word_in_brand'] = df['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))

        #just keep features
        df.drop(['id', 'product_title', 'search_term', 'product_description',             'brand', 'complete_product_desc', 'attr'], axis=1, inplace=True)
        
        return df


    feature_df=train_df.copy()


    feature_df= features(feature_df)


    # - testing by running model for rank
    # - Also, using rank as a feature to boost relevance prediction.

    # Getting the features for training


    feature_relev=feature_df.copy()
    feature_rank=feature_df.copy()
    feature_relev=feature_relev.drop(['rank'],axis=1)
    feature_rank=feature_rank.drop(['relevance'],axis=1)


    #need to work on relevance and rank as one, so using rank as a feature for now(can remove later) to test
    X = feature_df.loc[:, feature_df.columns != 'relevance'] 
    y = feature_df.loc[:, feature_df.columns == 'relevance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    # XGBoost training


    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
    xgb.fit(X_train, y_train.values.ravel())
    y_pred_x = xgb.predict(X_test)
    xgb_mse = mean_squared_error(y_pred_x, y_test)
    xgb_rmse = np.sqrt(xgb_mse)
    print('Xgboost RMSE: %.4f' % xgb_rmse)


    # ***test data***

    test_df=pd.read_csv('data/test_new.csv').dropna().reset_index().drop(['index'],axis=1)


    predicted_df=test_df.copy()

    predicted_df=predicted_df[['product_title','_unit_id','query','product_description','brand']]


    test_df=test_df[['Unnamed: 0', '_unit_id','relevance','product_title','query','rank','product_description','brand']]
    test_df.rename(columns={'Unnamed: 0':'id', '_unit_id':'product_uid','query':'search_term'}, inplace=True)
    test_df['product_description'] = test_df['product_description'].replace(np.nan, '', regex=True)
    test_df['relevance'] = test_df['relevance'].replace(np.nan, 0, regex=True)
    test_df.head(5)

    test_features=features(test_df)
    test_features.columns


    test_X = test_features.loc[:, test_features.columns != 'relevance'] 
    test_y = test_features.loc[:, test_features.columns == 'relevance']

    # predicting on test dataset

    test_pred = xgb.predict(test_X)
    test_xgb_mse = mean_squared_error(test_pred, test_y)
    test_xgb_rmse = np.sqrt(xgb_mse)
    print('Xgboost RMSE: %.4f' % test_xgb_rmse)


    predicted_df['predicted_relevance']=test_pred
    print(predicted_df.head(5))


    predicted_df=predicted_df.sort_values('predicted_relevance', ascending=False)

    print(predicted_df.head(5))

    gr = predicted_df.groupby('query')['predicted_relevance'].apply(list).reset_index(name='predicted_relevance')

    pid = predicted_df.groupby('query')['_unit_id'].apply(list).reset_index(name='product_uid')

    merged = gr.merge(pid,how='left',on='query')


    print(merged.head(5))


    from data_utils import *
    def gModelRef(compare_dict):

    refer_dict = {}
    with open('data/reference/s_dict_test.txt') as lines:
        for line in lines:
            split_line = line.split('||')
            refer_dict[split_line[0]] = int(split_line[1].replace('\n',''))

    new_dict = {}

    for k,v in compare_dict.items():
        if refer_dict.get(" ".join(cleanData(k))):
            new_dict[refer_dict[" ".join(cleanData(k))]] = v

    return new_dict


    def dictBaseline(mergedbs):
    final_dict = {}
    
    for i in range(len(mergedbs)):
        argsorted = np.argsort(-np.array(mergedbs['predicted_relevance'][i]))
        app_lst = []

        
        final_dict[str(mergedbs['query'][i])] = {}

        for j in range(len(argsorted)):
            final_dict[str(mergedbs['query'][i])][str(mergedbs['product_uid'][i][j] )] = str(argsorted[j])

    return final_dict

    def makeOutFile(op_dict,op_file):
        if not os.path.exists('output'):
            os.mkdir('output')
            
        with open('output/'+op_file,'w') as ft:
            for k,v in op_dict.items():
                app_lst = []

                for sk,sv in v.items():
                    app_lst.append( str(sk) + '-' + str(sv) )

                ft.write( str(k) + ' ' + ",".join(app_lst) + '\n')


    makeOutFile(gModelRef(dictBaseline(merged)),'test_bs.out')


if __name__ == '__main__':
    main()