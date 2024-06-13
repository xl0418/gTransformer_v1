# Description: This file contains the code for creating a custom dataset class for PyTorch.

from torch.utils.data import Dataset
from BPEgenome import genomeBPE
import torch


class grDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, data, seq_len):
        'Initialization'
        self.labels = data[data['genomeid'].isin(list_IDs)]['score'].values
        self.list_IDs = list_IDs
        self.data = data
        self.seq_len = seq_len

        self.genome_bpe = genomeBPE(data['seq'].values)
        self.genome_bpe.prepare_token_vocab(num_tokens=5000, num_iterations=6000,
                                            save_vocab='data/tokens_frequencies.json',
                                            save_pretokenzation='data/vocab.json')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        seq_train = self.data[self.data['genomeid'] == ID]['seq'].values[0]
        gr_train = self.labels[index]

        # future work: mask, pad, and truncate sequences -- 06.11.2024

        # Tokenize the sequence
        seq_tokens_train = self.genome_bpe.tokenize_word(seq_train)
        seq_tokens_id = self.genome_bpe.token2id(seq_tokens_train)
        # Add padding to each sequence ids
        seq_tokens_padding_id = self.seq_len - len(seq_tokens_id)  # We will add [PAD]

        if seq_tokens_padding_id < 0:
            raise ValueError("Sentence is too long")

        seq_tokens_id = torch.concat(
            [torch.tensor(seq_tokens_id, dtype=torch.int64), torch.zeros(seq_tokens_padding_id, dtype=torch.int64)])

        assert len(seq_tokens_id) == self.seq_len, f"Error: len(seq_tokens_id) != len(seq_tokens_train) for ID: {ID}"

        return seq_tokens_id, gr_train

# if __name__ == '__main__':
#     # Test the Dataset class
#     data = pd.read_csv('data/GR3_ribosomal_maxg_expanded.tsv', sep='\t')
#     list_IDs = data['genomeid'].values
#     dataset = Dataset(list_IDs, data)
#     print(dataset.__len__())
#     print(dataset.__getitem__(2))
