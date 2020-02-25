from torch.utils.data import Dataset
import pandas as pd

class EcomDataSet(Dataset):

  def __init__(self, csv_file):
    self.pandas_df = pd.read_csv(csv_file)
    puidgp = self.pandas_df.groupby('search_term')['product_uid'].apply(list).reset_index(name='product_uid')
    ptgp = self.pandas_df.groupby('search_term')['product_title'].apply(list).reset_index(name='product_title')
    rlgp = self.pandas_df.groupby('search_term')['relevance'].apply(list).reset_index(name='relevance')
    brgp = self.pandas_df.groupby('search_term')['brand'].apply(list).reset_index(name='brand')
    pdgp = self.pandas_df.groupby('search_term')['product_description'].apply(list).reset_index(name='product_description')

    m0pd = puidgp.merge(ptgp,how='left',on='search_term')
    m1pd = m0pd.merge(rlgp,how='left',on='search_term')
    m2pd = m1pd.merge(brgp,how='left',on='search_term')
    self.pandas_df = m2pd.merge(pdgp,how='left',on='search_term')

  def __len__(self):
    return len(self.pandas_df)

  def __getitem__(self, idx):

        sample = {
            'query': self.pandas_df['search_term'],
            'product_title': [c for c in self.pandas_df['product_title']],
            'relevance': [c for c in self.pandas_df['relevance']],
            'brand': [c for c in self.pandas_df['brand']],
            'product_description': [c for c in self.pandas_df['product_description']]
            }
        

        return sample