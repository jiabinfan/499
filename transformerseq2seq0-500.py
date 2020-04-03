# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

import collections
from collections import Counter
import math
import numpy as np
import subprocess

# from google.colab import drive
# drive.mount('/content/drive')
#
# !python -m spacy download en
# !python -m spacy download fr
spacy_en = spacy.load('en')
spacy_de = spacy.load('fr')

# preparing data

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# spacy_de = spacy.load('de')

def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def compute_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(SRC, TRG))
print(type(test_data), len(test_data))
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print('Training on CPU')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, seq len, seq len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, seq len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, seq len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, seq len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, seq len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        # trg_pad_mask = [batch size, 1, trg len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention


# training

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]
            #            print(output.shape, output[1,:,:])
            output = output.contiguous().view(-1, output_dim)
            #            print(output.shape, output)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs

# N_EPOCHS = 20
# CLIP = 1

# best_valid_loss = float('inf')

# for epoch in range(N_EPOCHS):

#     start_time = time.time()

#     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#     valid_loss = evaluate(model, valid_iterator, criterion)

#     end_time = time.time()

#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)

#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), '/content/drive/My Drive/499/transformer_seq2seq.pt')

#     print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# model.load_state_dict(torch.load('/home/jiabin/projects/def-lilimou/jiabin/499/transformer_seq2seq.pt', map_location={'cuda:0': 'cpu'}))
model.load_state_dict(torch.load('transformer_seq2seq.pt', map_location='cpu'))

# /Users/jiabinfan/Documents/499
test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(model):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    # action_probs = np.random.rand(len(board.availables))
    # return zip(board.availables, action_probs)
    return 0


def policy_value_fn(state, model):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state
    temp reduce from 10 to 1 as time goes by
    """
    # return uniform probabilities and 0 score for pure MCTS
    # action_probs = np.ones(len(board.availables))/len(board.availables)
    # return zip(board.availables, action_probs), 0
    trg_tensor, enc_src, trg_mask, src_mask, trg_indexes = state[0], state[1], state[2], state[3], state[4]
    len_trg_indexes = len(trg_indexes)
    temp = 10 * 0.9 ** len_trg_indexes
    trg_mask = model.make_trg_mask(trg_tensor)
    output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
    output = output / temp
    action_probs = output.softmax(dim=2)[0][-1]

    action = [i for i in range(OUTPUT_DIM)]

    return zip(action, action_probs), 0


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """

        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value + self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def get_prior(self):
        return self._P

    def get_parent(self):
        return self._parent

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, device, model, policy_value_fn, trg_field, c_puct=5, n_playout=10):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._model = model
        self._device = device
        self._trg_field = trg_field

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        trg_iddd = None
        end_token = False
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            end_token = False
            action, node = node.select(self._c_puct)
            # do move
            trg_tensor, enc_src, trg_mask, src_mask, trg_indexes = state[0], state[1], state[2], state[3], state[4]
            trg_indexes.append(action)
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self._device)
            trg_mask = self._model.make_trg_mask(trg_tensor)
            state = (trg_tensor, enc_src, trg_mask, src_mask, trg_indexes)
            trg_iddd = trg_indexes
            if trg_indexes[-2] == self._trg_field.vocab.stoi[self._trg_field.eos_token]:
                end_token = True
        # print("play out branch trg_indexes: ", trg_iddd)
        action_probs, _ = self._policy(state, self._model)
        if end_token == False:
            node.expand(action_probs)
            # Evaluate the leaf node by random rollout
            leaf_value = self._evaluate_rollout(state)
            # Update value and visit count of nodes in this traversal.
            node.update_recursive(leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        # player = state.get_current_player()
        # for i in range(limit):
        #     end, winner = state.game_end()
        #     if end:
        #         break
        #     action_probs = rollout_policy_fn(state)
        #     max_action = max(action_probs, key=itemgetter(1))[0]
        #     state.do_move(max_action)
        # else:
        #     # If no break from the loop, issue a warning.
        #     print("WARNING: rollout reached move limit")
        # if winner == -1:  # tie
        #     return 0
        # else:
        #     return 1 if winner == player else -1
        avg_prior = 0
        current_node = self._root
        n = 0
        while current_node != None:
            prior = current_node.get_prior()
            avg_prior += prior
            n += 1
            current_node = current_node.get_parent()

        reward = avg_prior / n
        return reward

    def get_word(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current state

        Return: the selected action
        """
        for n in range(self._n_playout):
            trg_tensor, enc_src, trg_mask, src_mask, trg_indexes = state[0], state[1], state[2], state[3], state[4]
            trg_tensor_copy, trg_mask_copy, trg_indexes_copy = trg_tensor.clone(), trg_mask.clone(), trg_indexes.copy()
            state_copy = (trg_tensor_copy, enc_src, trg_mask_copy, src_mask, trg_indexes_copy)
            self._playout(state_copy)

        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSGenerator(object):
    """AI Generator based on MCTS"""

    def __init__(self, device, model, trg_field, c_puct=5, n_playout=2000):
        self.model = model
        self.mcts = MCTS(device, model, policy_value_fn, trg_field, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(1)

    def get_action(self, state):
        word = self.mcts.get_word(state)
        self.mcts.update_with_move(1)
        return word

    def __str__(self):
        return "MCTS {}".format(self.player)


def translate_sentence(sentence, src_field, trg_field, model, device, mcts_generator, max_len=50):
    model.eval()
    # tokenize the source sentence if it has not been tokenized (is a string)
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # append the <sos> and <eos> tokens
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    # numericalize the source sentence
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    # convert it to a tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # create the source sentence mask
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():

            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        state = (trg_tensor, enc_src, trg_mask, src_mask, trg_indexes)
        pred_token = mcts_generator.get_action(state)
        trg_indexes.append(pred_token)

        print("mcts pred_token: ", pred_token)
        print("mcts trg_indexes: ", trg_indexes)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    print("mcts index: ", trg_indexes)
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention


def greedy_translate_sentence(sentence, src_field, trg_field, model, device, mcts_generator, max_len=50):
    model.eval()
    # tokenize the source sentence if it has not been tokenized (is a string)
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # append the <sos> and <eos> tokens
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    # numericalize the source sentence
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    # convert it to a tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # create the source sentence mask
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():

            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    print("greedy indexes: ", trg_indexes)
    return trg_tokens[1:], attention


example_idx = 4

# model.load_state_dict(torch.load('/Users/jiabinfan/Documents/499/transformer_seq2seq.pt', map_location={'cuda:0': 'cpu'}))

# model.load_state_dict(torch.load('/home/jiabin/projects/def-lilimou/jiabin/499/transformer_seq2seq.pt', map_location={'cuda:0': 'cpu'}))
src = vars(train_data.examples[example_idx])['src']
trg = vars(train_data.examples[example_idx])['trg']

# print(f'src = {src}')
# print(f'trg = {trg}')
mcts_generator = MCTSGenerator(device, model, TRG, c_puct=5, n_playout=200)
# greedy_translation, attention = greedy_translate_sentence(src, SRC, TRG, model, device, mcts_generator)
# translation, attention = translate_sentence(src, SRC, TRG, model, device, mcts_generator)
# mcts_bleu = compute_bleu( trg, translation)
# greedy_bleu = compute_bleu(trg, greedy_translation)
# print(f'src = {src}')
# print(f'trg = {trg}')
# print("mcts translation: ", translation)
# print("greedy translation: ", greedy_translation)
# print("bleu for greedy: ",  greedy_bleu)
# print("bleu for mcts: ", mcts_bleu)

sum_greedy_bleu = 0
sum_mcts_bleu = 0
for i in range(500):
    src = vars(test_data.examples[i])['src']
    trg = vars(test_data.examples[i])['trg']
    greedy_translation, attention = greedy_translate_sentence(src, SRC, TRG, model, device, mcts_generator)

    mcts_translation, attention = translate_sentence(src, SRC, TRG, model, device, mcts_generator)

    sentence = open("sentenceCPU0-500.txt", 'a')
    sentence.write("\nmcts: " + str(mcts_translation))
    sentence.write("\ngreedy: " + str(greedy_translation))
    sentence.write("\ngt: " + str(trg) + '\n')

    greedy_bleu = compute_bleu(greedy_translation, trg)
    sum_greedy_bleu += greedy_bleu

    mcts_bleu = compute_bleu(mcts_translation, trg)
    sum_mcts_bleu += mcts_bleu

    mcts_file = open("mcts_bleuCPU0-500.txt", 'a')
    mcts_file.write(str(mcts_bleu) + '\n')
    greedy_file = open("greedy_bleuCPU0-500.txt", 'a')
    greedy_file.write(str(greedy_bleu) + '\n')

avg_greedy_bleu = sum_greedy_bleu / len(test_data)
avg_mcts_bleu = sum_mcts_bleu / len(test_data)
print("greedy bleu : ", avg_greedy_bleu)
print("mcts bleu: ", avg_mcts_bleu)