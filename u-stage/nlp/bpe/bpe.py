from typing import List, Dict, Set
from itertools import chain
import re
from collections import defaultdict, Counter


class BytePairEncoding(object):
    """ Byte Pair Encoding class
    We aren't gonna use this class for encoding. Because it is too slow......
    We will use sentence piece Google have made.
    Thus, this class is just for special token index reference.
    """
    PAD_token = '<pad>'
    PAD_token_idx = 0
    UNK_token = '<unk>'
    UNK_token_idx = 1
    CLS_token = '<cls>'
    CLS_token_idx = 2
    SEP_token = '<sep>'
    SEP_token_idx = 3
    MSK_token = '<msk>'
    MSK_token_idx = 4

    WORD_END = '_'

    def __init__(self, corpus: List[List[str]], max_vocab_size: int) -> None:
        self.idx2word = build_bpe(corpus, max_vocab_size)

    def encode(self, sentence: List[str]) -> List[int]:
        return encode(sentence, self.idx2word)

    def decoder(self, tokens: List[int]) -> List[str]:
        return decode(tokens, self.idx2word)


def build_bpe(
        corpus: List[str],
        max_vocab_size: int
) -> List[int]:
    """ BPE Vocabulary Builder
    Implement vocabulary builder for byte pair encoding.
    Please sort your idx2word by subword length in descending manner.

    Hint: Counter in collection library would be helpful

    Note: If you convert sentences list to word frequence dictionary,
          building speed is enhanced significantly because duplicated words are
          preprocessed together

    Arguments:
    corpus -- List of words to build vocab
    max_vocab_size -- The maximum size of vocab

    Return:
    idx2word -- Subword list
    """
    # Special tokens
    PAD = BytePairEncoding.PAD_token  # Index of <PAD> must be 0
    UNK = BytePairEncoding.UNK_token  # Index of <UNK> must be 1
    CLS = BytePairEncoding.CLS_token  # Index of <CLS> must be 2
    SEP = BytePairEncoding.SEP_token  # Index of <SEP> must be 3
    MSK = BytePairEncoding.MSK_token  # Index of <MSK> must be 4
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]

    WORD_END = BytePairEncoding.WORD_END  # Use this token as the end of a word
    # YOUR CODE HERE
    # 1. character vocabulary로 symbol vocab 초기화하고 단어를 sequence of chars로 표현
    vocab = {" ".join(list(word) + [WORD_END]): ct for word, ct in Counter(corpus).items()}
    chars = list(set([char for word in corpus for char in word]))
    num_merges = max_vocab_size - len(SPECIAL) - 1 - len(chars)
    # 2. number of merge operation에 도달할 때 까지 아래 두 과정을 반복한다
    for _ in range(num_merges):
        # 2-a. symbol pair를 센다. 합칠 pair가 없다면 loop을 종료한다.
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i],symbols[i+1]] += freq
        if not pairs:
            break
        # 2-b. 가장 빈번히 등장하는 pairs를 합쳐 새로운 symbol로 대체한다
        best = max(pairs, key=pairs.get)
        new_vocab = {}
        bigram = re.escape(' '.join(best))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            w_out = p.sub(''.join(best), word)
            new_vocab[w_out] = vocab[word]
        vocab = new_vocab
        chars.append(''.join(best))
    idx2word = SPECIAL + sorted(chars, key=len, reverse=True) + [WORD_END]
    return idx2word
