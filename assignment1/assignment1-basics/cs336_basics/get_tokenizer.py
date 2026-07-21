import os
import regex as re
from typing import List,Iterable
from collections import Counter


def train_bpe1(input_path: str | os.PathLike,vocab_size:int,special_tokens:List[str],**kwargs):
    with open(input_path,'r') as f:
        text=f.read()

    pat=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_pattern='|'.join(re.escape(tok) for tok in special_tokens)
    parts=re.split(special_pattern,text)
    words=[]
    for part in parts:
        word=re.findall(pat,part)
        words.extend(word)

    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])

    for token in special_tokens:
        vocab[len(vocab)]=token.encode('utf-8')

    word_freq=Counter()
    for word in words:
        tokens=[bytes([i]) for i in word.encode('utf-8')]
        word_freq[tuple(tokens)]+=1

    merges=[]
    while len(vocab)<vocab_size:
        pairs=Counter()
        for tokens,freq in word_freq.items():
            for i in range(len(tokens)-1):
                pairs[(tokens[i],tokens[i+1])]+=freq

        best_pair=max(pairs,key=lambda x:(pairs[x],x))
        merges.append(best_pair)
        new_token=best_pair[0]+best_pair[1]
        vocab[len(vocab)]=new_token

        new_word_freq=Counter()
        for tokens,freq in word_freq.items():
            i=0
            new_tokens=[]
            while i<len(tokens):
                if i <len(tokens)-1 and (tokens[i],tokens[i+1])==best_pair:
                    new_tokens.append(tokens[i]+tokens[i+1])
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            new_word_freq[tuple(new_tokens)]+=freq

        word_freq=new_word_freq

    return vocab,merges





class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:List[str]):
        self.vocab=vocab
        # reverse 的 key 是 bytes；查 special 时要用 part.encode("utf-8")，不能用裸 str
        self.reverse_vocab={v:k for k,v in self.vocab.items()}
        self.merges=merges
        # adapter 可能传 None，要 or []
        self.special_tokens=special_tokens or []
        # 错过1：按列表原顺序拼 A|B 时，短的 <|endoftext|> 会先匹配，
        # 长的 <|endoftext|><|endoftext|> 被拆成两个 → overlapping 测试 count 变成 3。
        # 必须按长度从长到短排序。
        # 错过2：不要用 list.sort(reverse=True)：
        #   - sort() 返回 None，join 会挂；
        #   - 默认按字典序不是按 len。
        # 要用 sorted(..., key=len, reverse=True)。
        sorted_special_tokens=sorted(self.special_tokens,key=len,reverse=True)
        if sorted_special_tokens:
            # 用捕获组，re.split 才会把 special 本身留在 parts 里
            self.special_pattern='(' + '|'.join(re.escape(tok) for tok in sorted_special_tokens) + ')'
        else:
            # 错过3：special 为空时若仍写成 '()'，re.split 会把文本切碎，
            # roundtrip 还能过，但 id 对不上 tiktoken（address/german matches 失败）。
            # 没有 special 时 pattern 设为 None，encode 里整段当普通文本。
            self.special_pattern=None
        self.regex=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def encode(self,text: str):
        # 先按特殊字符分块（无 special 则不 split）
        if self.special_pattern:
            parts=re.split(self.special_pattern,text)
        else:
            parts=[text]
        tokens=[]
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                # 错过：reverse_vocab[part]（str）→ KeyError；键是 bytes
                tokens.append(self.reverse_vocab[part.encode('utf-8')])
            else:
                words=re.findall(self.regex,part)
                for word in words:
                    # 单字节 bytes 序列，不要 list(encode) 得到 int
                    bytes_word=[bytes([b]) for b in word.encode('utf-8')]
                    # 按 merges 学习顺序逐条应用；每条 merge 扫完整段后再换下一条。
                    # 错过：曾把 tokens.extend 写在内层、或按位置 i 套 merges，
                    # 会导致同一段被重复 append / merge 顺序错误。
                    for merge in self.merges:
                        i=0
                        new_bytes_word=[]
                        while i<len(bytes_word):
                            if i<len(bytes_word)-1 and (bytes_word[i],bytes_word[i+1])==merge:
                                new_bytes_word.append(merge[0]+merge[1])  # bytes 拼接
                                i+=2
                            else:
                                new_bytes_word.append(bytes_word[i])
                                i+=1
                        bytes_word=new_bytes_word
                    # 该 pre-token 全部 merge 完后再转 id
                    tokens.extend([self.reverse_vocab[i] for i in bytes_word])

        return tokens

    def decode(self,tokens: List[int]):
        # id → bytes → 先 join 成一整段再 decode（不要逐段 decode 再拼 str，
        # 多字节字符可能跨 token）
        bytes_tokens=[self.vocab[i] for i in tokens]
        bytes_text=b''.join(bytes_tokens)
        text=bytes_text.decode('utf-8',errors='replace')
        return text

    def encode_iterable(self,iterable: Iterable[str]): #iterable指的是一个可迭代的对象，比如一个列表或一个生成器，这里可以使用文件对象
        for chunk in iterable: #遍历iterable中的每个元素，这里表示文件的每一行
            for token_id in self.encode(chunk): #对每个元素进行编码
                yield token_id #逐个yield，而不是一次性返回所有token_id



