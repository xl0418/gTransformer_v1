# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings


warnings.simplefilter("ignore")
print(torch.__version__)


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

    def forward(self, x):
        return self.embedding(x)


class PositionEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.seq_len = seq_len

        pe = torch.zeros(seq_len, self.embedding_size)

        for i in range(seq_len):
            for pos in range(0, self.embedding_size, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * pos) / self.embedding_size)))
                pe[i, pos + 1] = math.cos(pos / (10000 ** ((2 * pos) / self.embedding_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :self.seq_len], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size = 768, n_heads = 8):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        assert self.embedding_size % self.n_heads == 0, "embedding size is not divisible by n_heads"

        self.single_embedding_size = self.embedding_size // self.n_heads #768 / 8 = 96

        ### key, query and value matrix
        self.k_matrix = nn.Linear(self.single_embedding_size, self.single_embedding_size) # in 96 out 96
        self.q_matrix = nn.Linear(self.single_embedding_size, self.single_embedding_size)
        self.v_matrix = nn.Linear(self.single_embedding_size, self.single_embedding_size)

        ### out matrix
        self.o = nn.Linear(self.n_heads * self.single_embedding_size, self.embedding_size)


    def forward(self, k, q, v, mask = None):
        batch_size = k.size(0)
        seq_len = k.size(1)

        seq_len_query = q.size(1)

        k = k.view(batch_size, seq_len, self.n_heads, self.single_embedding_size) ## batch X seq_len X 8 X 96
        q = q.view(batch_size, seq_len_query, self.n_heads, self.single_embedding_size)
        v = v.view(batch_size, seq_len, self.n_heads, self.single_embedding_size)

        k = self.k_matrix(k) # batch X seq_len X 8 X 96  multiply with 96 X 96 (self.k_matrix) = batch X seq_len X 8 X 96
        q = self.q_matrix(q)
        v = self.v_matrix(v)

        k = k.transpose(1, 2)  # batch X 8 X seq_len X 96
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute attention score
        k_adjusted = k.transpose(-1, -2) # batch X 8 X 96 X seq_len
        product = torch.matmul(q, k_adjusted) # (batch X 8 X seq_len X 96)  X (batch X 8 X 96 X seq_len) = batch X 8 X seq_len X seq_len

        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / math.sqrt(self.single_embedding_size)

        scores = F.softmax(product, dim = -1)

        scores = torch.matmul(scores, v) # (batch X 8 X seq_len X seq_len) X (batch X 8 X seq_len X 96) = batch X 8 X seq_len X 96

        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_len_query, self.single_embedding_size*self.n_heads)

        output = self.o(concat)

        return output


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, n_heads = 8, expansion_factor = 4):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_size, n_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, expansion_factor * embedding_size),
            nn.ReLU(),
            nn.Linear(expansion_factor*embedding_size, embedding_size)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, k, q, v):
        attention_out = self.attention(k, q, v)
        attention_residual_out = attention_out + v # add & norm laryer: add
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) # add & norm layer: norm

        feed_fwd_out = self.feed_forward(norm1_out)

        feed_fwd_residual_out = feed_fwd_out + norm1_out # add
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embedding_size, num_encoderblocks = 2, expansion_factor = 4, n_heads = 8):
        super().__init__()

        self.embedding_layer = Embedding(vocab_size, embedding_size)
        self.position_layer = PositionEmbedding(seq_len, embedding_size)

        self.layers = nn.ModuleList([EncoderBlock(embedding_size, expansion_factor, n_heads) for _ in range(num_encoderblocks)])
        self.denselayer1 = nn.Linear(seq_len * embedding_size, embedding_size)
        self.denselayer2 = nn.Linear(embedding_size, 1)

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.position_layer(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)

        out = out.view(out.size(0), -1)
        out = self.denselayer1(out)
        # ReLU activation function
        out = F.relu(out)
        out = self.denselayer2(out)
        out = F.relu(out)


        return out # should be batch_size * 1 --> predictions of max growth rates


class Transformer(nn.Module):
    def __init__(self, seq_len, vocab_size, embedding_size, num_encoderblocks = 2, expansion_factor = 4, n_heads = 8):
        super().__init__()

        self.encoder = TransformerEncoder(seq_len, vocab_size, embedding_size, num_encoderblocks, expansion_factor, n_heads)

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    seq = torch.randint(0, 5005, (8, 256))
    model = Transformer(256, 5006, 768, 2, 4, 8)
    out = model(seq)

    ######## check tensor shapes ########
    embedder = Embedding(5006, 768)
    after_embed = embedder(seq)
    print(after_embed.shape)

    pos_embedder = PositionEmbedding(256, 768)
    after_pos_embed = pos_embedder(after_embed)
    print(after_pos_embed.shape)

    layers = nn.ModuleList(
        [EncoderBlock(768, 4, 8) for _ in range(2)])
    for layer in layers:
        after_pos_embed = layer(after_pos_embed, after_pos_embed, after_pos_embed)
        print(after_pos_embed.shape)

    out = after_pos_embed.view(after_pos_embed.size(0), -1)
    denselayer1 = nn.Linear(256 * 768, 768)
    out = denselayer1(out)
    out = F.relu(out)
    print(out.shape)
    denselayer2 = nn.Linear(768, 1)


    out = denselayer2(out)
    print(out.shape)

    def count_params(model, is_human: bool = False):
        params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f"{params / 1e6:.2f}M" if is_human else params

    print(model)
    print("Total # of params:", count_params(model, is_human=True))


    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)