from torch.utils.data import Dataset
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer,
                      Linear, Embedding)
import datasets
import torch
import torch.nn as nn
import math


class RNNDataset(Dataset):
    def __init__(self,
                 dataset: datasets.arrow_dataset.Dataset,
                 max_seq_length: int
                ):
        self.max_seq_length = max_seq_length + 2 #Because we are adding the start and stop tokens
        self.dataset = dataset
        
        dataset_vocab = self.get_dataset_vocabulary()
        # defining a dictionary that simply maps tokens to their respective index in the embedding matrix
        self.word_to_index = {word: idx for idx,word in enumerate(dataset_vocab)}
        self.index_to_word = {idx: word for idx,word in enumerate(dataset_vocab)}
        
        self.pad_idx = self.word_to_index["<pad>"]
        
        self.preprocessed_dataset = []
        for ex in self.dataset :
            self.preprocessed_dataset.append('<start> '+ex['text']+' <stop>')

    def __len__(self):
        return len(self.preprocessed_dataset)
    
    def __getitem__(self, idx):
        tokens = self.preprocessed_dataset[idx].split()
        tokens_ids = [self.word_to_index.get(tok, self.word_to_index.get('<unk>')) for tok in tokens]
        tokens_ids_padded = tokens_ids + [self.pad_idx] * (self.max_seq_length -len(tokens_ids))
        return torch.tensor(tokens_ids_padded)
    
    def get_dataset_vocabulary(self):
        vocab = sorted(set(" ".join([sample["text"] for sample in self.dataset]).split()))
        # we also add a <start> token to include initial tokens in the sentences in the dataset
        vocab += ["<start>", '<stop>', '<pad>']
        return vocab
    


class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, 
                 embedding_dim,
                 hidden_dim,
                 num_layers, 
                 dropout_rate,
                 embedding_weights=None,
                 freeze_embeddings=False):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # pass embeeding weights if exist
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weights,
                                                          freeze = freeze_embeddings)

        else:  # train from scratch embeddings
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = torch.nn.LSTM(input_size=embedding_dim,    # The number of expected features in the input x
                           hidden_size=hidden_dim,  # The size of the hidden state vector h
                           num_layers=num_layers,   # Number of recurrent layers. E.g., setting num_layers=2 would stack two RNNs together
                           batch_first=True,  
                      )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_id):
        embedded = self.embedding(input_id)
        embedded = self.dropout(embedded)
        outputs, hidden = self.lstm(embedded)
        predictions = self.fc(outputs[:, :-1, :]).permute(0, 2, 1)
        targets = input_id[:, 1:] # We want to predict the next words
        return predictions, targets

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        token_with_positional_embedding = token_embedding + self.pos_embedding[:token_embedding.size(0),:]
        token_with_positional_embedding = self.dropout(token_with_positional_embedding)
        return token_with_positional_embedding

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

DROPOUT = 0.01
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
PAD_VALUE = 0
        
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        # YOUR CODE HERE
        self.hidden_size = hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.tok_emb = TokenEmbedding(input_vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=DROPOUT)
        
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=NHEAD,
                                                dim_feedforward=hidden_size)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)

        decoder_layer = TransformerDecoderLayer(d_model=hidden_size, nhead=NHEAD,
                                                dim_feedforward=hidden_size)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=NUM_DECODER_LAYERS)

        self.generator = nn.Linear(hidden_size, output_vocab_size)
    
    # Calculate the square subsequent mask as a upper triangular matrix
    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def create_mask(src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = EncoderDecoder.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

        src_padding_mask = (src == PAD_VALUE).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_VALUE).transpose(0, 1)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
        
    def forward(self, inputs, targets= None): 
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = EncoderDecoder.create_mask(inputs, targets)
        src_emb = self.positional_encoding(self.tok_emb(inputs))
        tgt_emb = self.positional_encoding(self.tok_emb(targets))
        memory = self.encoder(src_emb, src_mask.to(self.device), src_padding_mask.to(self.device))
        outs = self.decoder(tgt_emb, memory, tgt_mask.to(self.device), None,
                                        tgt_padding_mask.to(self.device), src_padding_mask.to(self.device))
        return self.generator(outs)