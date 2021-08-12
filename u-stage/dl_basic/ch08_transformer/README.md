# 8강 Sequential Models - Transformer

- **Sequential model의 한계점과** 이를 해결 하기 위해 등장한 **Transformer에** 대해 배운다
- Transformer는 Encoder와 Decoder로 구성되어 있지만, 강의에서는 **Encoder와 Multi-Head Attention에 대해 좀 더 집중적으로** 배우자

[back to super](https://github.com/jinmang2/BoostCamp_AI_Tech_2/tree/main/u-stage/dl_basic)

## Transformer
- 강의에선 아래 자료를 기반으로 transformer의 구조를 다룸
    - [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
- 정말 다양한 Transformer에 대한 자료가 존재합니다!!
    - [https://youtu.be/KQfvEg-fGMw](https://youtu.be/KQfvEg-fGMw)
    - [https://youtu.be/x_8cp4Vdnak](https://youtu.be/x_8cp4Vdnak)
    - [https://youtu.be/xhY7m8QVKjo](https://youtu.be/xhY7m8QVKjo)
    - [https://youtu.be/AA621UofTUA](https://youtu.be/AA621UofTUA)
    - [https://nlp.seas.harvard.edu/2018/04/03/attention.html](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

## Self-Attention Code Script

```python
import math
import torch
import torch.nn as nn

vocabulary = {
    "[BOS]": 0,
    "[PAD]": 1,
    "[UNK]": 3,
    "#나": 172,
    "는": 435,
    "#경찰": 4623,
    "서": 457,
    "에": 774,		
    "#갔": 577,		
    "다": 12,
}

# tokenize and encode
# 여기서는 tokenize가 아래와 같이 수행됐다고 가정
tokenized = ["#나", "는", "#경찰", "서", "에", "#갔", "다",]
encodes = [vocabulary.get(token, vocabulary["[UNK]"]) for token in tokenized]
input_ids = torch.LongTensor([encoded]) # (bsz, seq_length)

# Embedding
# BERT 이후의 논문부터는 embed_dim == hidden_dim으로 맞춰준다.
embedding = nn.Embedding(30000, 128) # (vocab_size, hid_dim)
x = embedding(input_ids) # (bsz, seq_length, hid_dim)

# attention mask 정의
batch_size, seq_length = input_idx.size()
attn_mask = input_ids.eq(vocabulary["[PAD]"])
attn_mask = attn_mask.unsqueeze(1).expand(batch_size, seq_length, seq_length)

# Self-Attention parameter 정의
hidden_dim = 128 # 시작인 embed_dim일 수도 있음
W_q = nn.Linear(hidden_dim, hidden_dim)
W_k = nn.Linear(hidden_dim, hidden_dim)
W_v = nn.Linear(hidden_dim, hidden_dim)
W_o = nn.Linear(hidden_dim, hidden_dim)

# projection
# self-attention이기 때문에 src_seq_length == tgt_seq_length임
Q = W_q(x) # (bsz, src_seq_length, hid_dim)
K = W_k(x) # (bsz, tgt_seq_length, hid_dim)
V = W_v(x) # (bsz, tgt_seq_length, hid_dim)

# Multi-Head split
num_heads = 8
head_dim = hidden_dim // num_heads
assert hidden_dim % num_heads == 0, "head의 수는 hidden_dim의 약수"

# Scaling
Q = Q / math.sqrt(hidden_dim // num_heads)

def _shape(tensor: torch.Tensor, seq_len: int, bsz: int):
    new_shape = (bsz, seq_len, num_heads, hidden_dim // head_dim)
    return tensor.view(*new_shape).transpose(1, 2).contiguous()

# BART style! 4차원으로 하나 3차원으로 하나 동일하다.
# (bsz*n_heads, seq_length, hid_dim//n_heads)
proj_shape = (batch_size* num_heads, -1, head_dim)
Qs = _shape(Q).view(*proj_shape)
Ks = _shape(K).view(*proj_shape)
Vs = _shape(V).view(*proj_shape)

# Calculate attention matrix
attn_scores = torch.bmm(Qs, Ks.transpose(1, 2))

# mask pad
attn_scores = attn_scores.masked_fill(attn_mask, -float("inf"))

# Calc softmax
attn_probs = torch.softmax(attn_scores, dim=-1)

# Matrix Multiplication between attn_scores and Vs
attn_output = torch.bmm(attn_probs, Vs)

# Calculate output
attn_output = attn_output.view(batch_size, num_heads, seq_length, head_dim)
attn_output = attn_output.transpose(1, 2)
attn_output = attn_output.reshape(batch_size, seq_length, hidden_dim)

attn_output = W_o(attn_output)
```

## Vision Transformer
- https://github.com/lucidrains/vit-pytorch


- NLP의 Transformer를 vision에도 적용!
    - related work를 읽어보니, 처음 시도는 아님
        - naive하게 self-attn을 cnn에 적용한 사례도 있었고
        - ViT와 동일한 연구도 있었으나 (2 X 2 patch로 extract) scale이 작았음
        - iGPT도 있었음! ViT가 성능 우수
- CNN 대비 inductive bias가 조금은 떨어지나, scale을 키워서 다소 해소!
    - Inductive Bias?
        - locality
        - 2-dimensional neighborhood structure
        - translation equivariance
    - ViT에서 MLP는 locality와 translation equivariance하지만 self-attention은 global하다고 함
        - quadratic한 문제가 동일하게 발생할 것 같음
    - 2-D 구조 positional embedding이 그닥 효과가 좋진 않았다고 함
- Transformer 중 `BERT`에서 많은 영감을 얻은 것으로 보임!
    - ImageNet 등 분류 문제를 풀기 위해 제시되서 Encoder 구조만으로도 충분했을 듯
    - BERT처럼 `[CLS]` 토큰을 제일 앞에 넣어서 fine-tune시 이 토큰을 활용하여 분류를 수행하도록 함
    - input image를 patch로 나눠 각 patch를 nlp의 token처럼 flatten하여 feeding해준다고 함
    - non-linearity로 `GELU`를 사용
    - `Pre-Layer Normalization` 구조 사용
        - 원래 transformer는 `LayerNorm(x + sublayer(x))`임
        - PreLN 구조는 `x + sublayer(LayerNorm(x))`임
- ResNet에 Group Normalization, Standardized convolution을 사용하여 resnet으로 supervised transfer learning을 하게 한 `Big Transfer` 대비 우수한 성능을 보임
- `Attention Distance`가 `Receptive Field Size`와 유사하다고 함
    - 향후 논문 더 자세히 뜯어보며 공부해야 할 듯...


## DALL-E
- https://openai.com/blog/dall-e/
