import torch
import torch.nn as nn
import torch.nn.functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CPSIR(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
                 dropout=0.2, window = 1, kernel_size = 1, batch_size = 1):

        super(CPSIR, self).__init__()

        self.context_window = window

        self.batch_size = batch_size
        
        self.word_embeddings = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size,padding_idx=1).to(device)

        self.Dropout = nn.Dropout(p=dropout).to(device)

        self.queryConv1d = nn.Conv1d(self.context_window * embedding_dim, hidden_dim, kernel_size).to(device)
        self.queryLinear = nn.Linear(hidden_dim, tagset_size).to(device)

        self.productConv1d = nn.Conv1d(self.context_window * embedding_dim, hidden_dim, kernel_size).to(device)
        self.productLinear = nn.Linear(hidden_dim, tagset_size).to(device)

    def sliding_window(self, seq):
        seq_length = seq.shape[1]
        assert seq_length >= self.context_window
        constituents = []
        offset = seq_length - self.context_window + 1
        for i in range(self.context_window):
            constituents.append(seq[:, i:offset + i, :])
        out = torch.cat(constituents, dim=-1).to(device)
        return out

    def forward(self, queries, products):
      
        queries  = queries.reshape(self.batch_size,queries.shape[0])
        products = products.reshape(self.batch_size,products.shape[0],products.shape[1])


        assert queries.shape[0] == products.shape[0]
        
        batch_size = queries.shape[0]
        qlen = products.shape[1]
        num_docs, dlen = products.shape[1], products.shape[2]

        
        queries_embeds = self.word_embeddings(queries)
        queries_embeds = self.Dropout(queries_embeds)
        queries_embeds = self.sliding_window(queries_embeds)
        query_rep = self.queryConv1d(queries_embeds.transpose(1, 2)).transpose(1, 2)
        query_rep = torch.tanh(self.queryLinear(torch.tanh(query_rep)))
        query_max_pool = query_rep.max(1)[0] 
        

        doc_rep = products.view(batch_size * num_docs, dlen)
        queries_products = self.word_embeddings(doc_rep)
        queries_products = self.Dropout(queries_products)
        queries_products = self.sliding_window(queries_products)
        doc_rep = self.productConv1d(queries_products.transpose(1, 2)).transpose(1, 2)
        doc_rep = torch.tanh(self.productLinear(torch.tanh(doc_rep)))
        product_max_pool = doc_rep.max(1)[0]
        product_max_pool = product_max_pool.view(batch_size, num_docs, -1)
        
        query_max_pool = query_max_pool.unsqueeze(1).expand(*product_max_pool.size())

        return f.cosine_similarity(query_max_pool, product_max_pool, dim=2)
        