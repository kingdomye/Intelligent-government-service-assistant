import torch
import torch.nn as nn
import math
from typing import List, Tuple
from transformers import BertTokenizerFast

MAX_LEN   = 256
D_MODEL   = 256
N_HEAD    = 8
N_LAYER   = 4
D_FF      = 1024
VOCAB     = 21128          # 中文 BERT 词表大小，可换成自己统计
PAD_ID    = 0

def char_span_to_token_span(offset, char_start, char_end):
    """offset: List[(token_start_char, token_end_char)]"""
    token_start = token_end = 0
    for idx, (s, e) in enumerate(offset):
        if s <= char_start < e:
            token_start = idx
        if s < char_end <= e:
            token_end = idx
            break
    return token_start, token_end

class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=MAX_LEN):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:x.size(1)]

def encode_qa_pair(tokenizer, question, context, answer_text):
    enc = tokenizer.encode_plus(
        question, context,
        max_length=MAX_LEN,
        truncation='only_second',
        return_offsets_mapping=True,
        padding='max_length',
    )
    offset = enc['offset_mapping']

    # 答案在 **原始上下文** 的字符区间
    ans_char_start = context.find(answer_text)
    ans_char_end   = ans_char_start + len(answer_text)
    if ans_char_start == -1:
        raise ValueError("答案不在上下文中！")

    # 把字符区间映射到 token 区间
    token_start = token_end = 0
    for idx, (s, e) in enumerate(offset):
        # offset 是整句的字符区间
        if s <= ans_char_start < e:
            token_start = idx
        if s < ans_char_end <= e:
            token_end = idx
            break

    return {
        'input_ids': torch.tensor(enc['input_ids']),
        'start_pos': torch.tensor(token_start),
        'end_pos'  : torch.tensor(token_end),
    }

class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_head):
            super().__init__()
            assert d_model % n_head == 0
            self.d_k = d_model // n_head
            self.n_head = n_head
            self.qkv = nn.Linear(d_model, d_model * 3)
            self.out = nn.Linear(d_model, d_model)

        def forward(self, x, mask=None):
            B, L, _ = x.size()
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(B, L, self.n_head, self.d_k).transpose(1, 2)
            k = k.view(B, L, self.n_head, self.d_k).transpose(1, 2)
            v = v.view(B, L, self.n_head, self.d_k).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores.masked_fill_(mask == 0, -1e9)
            attn = torch.softmax(scores, dim=-1)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
            return self.out(out)

class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )

        def forward(self, x):
            return self.net(x)

class EncoderBlock(nn.Module):
        def __init__(self, d_model, n_head, d_ff):
            super().__init__()
            self.attn = MultiHeadAttention(d_model, n_head)
            self.ff = FeedForward(d_model, d_ff)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x, mask=None):
            x = self.norm1(x + self.attn(x, mask))
            x = self.norm2(x + self.ff(x))
            return x

class TransformerEncoder(nn.Module):
        def __init__(self, vocab, d_model, n_head, n_layer, d_ff):
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model)
            self.pe = PositionalEncoding(d_model)
            self.layers = nn.ModuleList([EncoderBlock(d_model, n_head, d_ff)
                                         for _ in range(n_layer)])

        def forward(self, x, mask=None):
            x = self.embed(x) * math.sqrt(x.size(-1))
            x = self.pe(x)
            for layer in self.layers:
                x = layer(x, mask)
            return x

class QAModel(nn.Module):
        def __init__(self, vocab=VOCAB):
            super().__init__()
            self.encoder = TransformerEncoder(vocab, D_MODEL, N_HEAD, N_LAYER, D_FF)
            self.start_head = nn.Linear(D_MODEL, 1)
            self.end_head = nn.Linear(D_MODEL, 1)

        def forward(self, input_ids):
            h = self.encoder(input_ids)  # [B, L, D]
            start_logits = self.start_head(h).squeeze(-1)
            end_logits = self.end_head(h).squeeze(-1)
            return start_logits, end_logits


def train_step(model, batch):
    input_ids, start_pos, end_pos = batch
    start_logits, end_logits = model(input_ids)
    loss = nn.CrossEntropyLoss()(start_logits, start_pos) + \
           nn.CrossEntropyLoss()(end_logits, end_pos)
    return loss

def predict(model, tokenizer, question, context):
    device = next(model.parameters()).device
    enc = tokenizer.encode_plus(
        question, context,
        max_length=256,
        truncation='only_second',
        return_offsets_mapping=True,
        padding='max_length',
    )
    input_ids = torch.tensor([enc['input_ids']]).to(device)
    sep_idx = enc['input_ids'].index(tokenizer.sep_token_id)  # 第一个 [SEP] 之后才是上下文

    model.eval()
    with torch.no_grad():
        start_logits, end_logits = model(input_ids)

    # 把问题部分（含[SEP]）屏蔽掉
    start_logits[0, :sep_idx + 1] = -1e9
    end_logits[0, :sep_idx + 1] = -1e9

    start = torch.argmax(start_logits[0]).item()
    end = torch.argmax(end_logits[0]).item()

    if start > end:
        return ""

    tokens = enc['input_ids'][start:end + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokenizer.decode(tokens, skip_special_tokens=True).replace(" ", "").strip()

tok = BertTokenizerFast.from_pretrained('bert-base-chinese')





