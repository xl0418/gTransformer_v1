import collections
import re
import pandas as pd
import json
import os
import tqdm


class genomeBPE():
    def __init__(self, data):
        self.data = data

    def get_vocab(self):
        vocab = collections.defaultdict(int)

        for seq in self.data:
            words = seq.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
        return vocab

    def get_stats(self):
        pairs = collections.defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        # Lookahead and Lookbehind: (?= … ), (?! … ), (?<= … ), (?<! … )  https://www.rexegg.com/regex-disambiguation.html#negative-lookbehind

        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            #  replace 'A A' with 'AA'
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def get_tokens_from_vocab(self, vocab):
        tokens_frequencies = collections.defaultdict(int)
        vocab_tokenization = {}
        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens_frequencies[token] += freq
            vocab_tokenization[''.join(word_tokens)] = word_tokens
        return tokens_frequencies, vocab_tokenization

    def measure_token_length(self, token):
        if token[-4:] == '</w>':
            return len(token[:-4]) + 1
        else:
            return len(token)

    def sorted_tokens_tuple(self, tokens_frequencies):
        return sorted(tokens_frequencies.items(), key=lambda item: (self.measure_token_length(item[0]), item[1]),
                      reverse=True)

    def prepare_token_vocab(self, num_tokens=5000, num_iterations=5000, save_vocab=None, save_pretokenzation=None):
        if os.path.isfile(save_vocab) and os.path.isfile(save_pretokenzation):
            print('Tokens and vocabulary files were detected...')
            print('Reading tokens and vocabulary from local files...')
            self.tokens_frequencies = json.load(open(save_vocab))
            self.vocab_tokenization = json.load(open(save_pretokenzation))
            self.sorted_tokens_tuple = sorted(self.tokens_frequencies.items(),
                                              key=lambda item: (self.measure_token_length(item[0]), item[1]),
                                              reverse=True)
            self.sorted_tokens = [token for (token, freq) in self.sorted_tokens_tuple]

            return
        else:
            print('No tokens and vocabulary files were detected...')
            print('Starting from preparing tokens and vocabulary...')
            print('==========')
            print('Tokens Before BPE')
            self.vocab = self.get_vocab()
            self.tokens_frequencies, self.vocab_tokenization = self.get_tokens_from_vocab(self.vocab)
            print('All tokens: {}'.format(self.tokens_frequencies.keys()))
            print('Number of tokens: {}'.format(len(self.tokens_frequencies.keys())))
            print('==========')

            i_token = 0
            i_iteration = 0

            while i_token < num_tokens and i_iteration < num_iterations:
                pairs = self.get_stats()
                if not pairs:
                    break
                best = max(pairs, key=pairs.get)
                self.vocab = self.merge_vocab(best, self.vocab)
                print('Iter: {}'.format(i_iteration))
                print('Best pair: {}'.format(best))
                self.tokens_frequencies, self.vocab_tokenization = self.get_tokens_from_vocab(self.vocab)
                print('All tokens: {}'.format(self.tokens_frequencies.keys()))
                print('Number of tokens: {}'.format(len(self.tokens_frequencies.keys())))
                print('==========')
                i_token += 1
                i_iteration += 1

            self.sorted_tokens_tuple = sorted(self.tokens_frequencies.items(),
                                              key=lambda item: (self.measure_token_length(item[0]), item[1]),
                                              reverse=True)
            self.sorted_tokens = [token for (token, freq) in self.sorted_tokens_tuple]

            # add '[PAD]' token
            self.sorted_tokens.append('[PAD]')
            self.tokens_frequencies['[PAD]'] = 1

        if save_vocab:
            # save self.tokens_frequencies
            json.dump(self.tokens_frequencies, open(save_vocab, 'w'))

        if save_pretokenzation:
            # save self.vocab_tokenization
            json.dump(self.vocab_tokenization, open(save_pretokenzation, 'w'))

    def tokenize_word(self, string, unknown_token='</u>'):
        if string == '':
            return []

        if self.sorted_tokens == []:
            return [unknown_token]

        string_tokens = []
        for i in range(len(self.sorted_tokens)):
            token = self.sorted_tokens[i]
            # token_reg = re.escape(token.replace('.', '[.]'))  #  no '.' in the sequences

            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token, string)]
            if len(matched_positions) == 0:
                continue
            substring_end_positions = [matched_position[0] for matched_position in matched_positions]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += self.tokenize_word(string=substring,
                                                    unknown_token=unknown_token)
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += self.tokenize_word(string=remaining_substring,
                                                unknown_token=unknown_token)
            break
        return string_tokens

    def token2id(self, string_tokens):
        vocab_list = list(self.tokens_frequencies.keys())
        return [vocab_list.index(x) for x in string_tokens]


if __name__ == '__main__':
    # read dna sequence from file
    dna_sequences = pd.read_csv('data/GR3_ribosomal_maxg_expanded.tsv', sep='\t')

    # len distribution of dna_sequences['dna']
    dna_sequences['len_gene'] = dna_sequences['dna'].apply(len)

    data = dna_sequences['dna'].values

    genome_bpe = genomeBPE(data)
    genome_bpe.prepare_token_vocab(num_tokens=5000, num_iterations=6000, save_vocab='data/tokens_frequencies.json',
                                   save_pretokenzation='data/vocab.json')

    # seq1 = 'ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC'
    # seq1_tokens = genome_bpe.tokenize_word(seq1)
    # seq1_tokens_id = genome_bpe.token2id(seq1_tokens)

    seq_len_list = []
    for seq in tqdm(data, desc='Tokenizing DNA sequences'):
        seq_tokens = genome_bpe.tokenize_word(seq)
        seq_tokens_id = genome_bpe.token2id(seq_tokens)
        seq_len_list.append(len(seq_tokens_id))

    print('Max seq length of GR3_ribosomal_maxg_expanded.tsv: {}'.format(max(seq_len_list)))
