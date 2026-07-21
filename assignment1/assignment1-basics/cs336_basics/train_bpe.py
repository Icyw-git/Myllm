import regex as re
import os
from collections import Counter

class BPE:
    def __init__(self,special_tokens: list[str]):
        self.special_tokens = special_tokens
        self.special_pattern='|'.join(re.escape(tok) for tok in self.special_tokens)
        # 用讲义给的 GPT-2 regex；不要用空白切词的白板简化版
        self.regex=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def train(self,input_path: str | os.PathLike,vocab_size: int,**kwargs):
        with open(input_path,'r') as f:
            text=f.read()
        # 训练前先按 special 切开，避免 special 参与 merge / 污染词表
        parts=re.split(self.special_pattern,text)
        words=[]
        for part in parts:
            words.extend(re.findall(self.regex,part))

        vocab={}
        for i in range(256):
            # 错过：写成 chr(i)（str）。测试要求 vocab value 是 bytes，应 bytes([i])
            vocab[i]=bytes([i])
        for token in self.special_tokens:
            # 错过：直接存 str。应 encode 成 bytes
            vocab[len(vocab)]=token.encode('utf-8')

        word_freq=Counter()
        for word in words:
            # 错过：list(word.encode(...)) 得到的是 int（如 32, 116），
            # merges 会变成 (32, 116) 而不是 (b' ', b't')；且 int 的 + 是加法不是拼接。
            # 正确：每个元素是长度为 1 的 bytes，如 bytes([b])
            tokens=[bytes([b]) for b in word.encode('utf-8')]
            word_freq[tuple(tokens)]+=1

        # 错过：把 merges=[] 放进 while 里，每轮清空，最后只剩 1 条 merge
        merges=[]

        while len(vocab)<vocab_size:
            pairs=Counter()

            for tokens,freq in word_freq.items():
                for i in range(len(tokens)-1):
                    pairs[(tokens[i],tokens[i+1])]+=freq
            if not pairs:
                break
            # 错过：max(pairs, key=pairs.get) —— 频次相同时只按遍历顺序，
            # 会和 reference 在某步分叉（如 index 64: (b'c',b'e') vs (b'l',b'e')）。
            # 平局时要按讲义对 pair 做字典序比较（这里用 (频次, pair)）。
            best_pair=max(pairs,key=lambda x: (pairs[x],x))
            # 这里是 bytes 拼接：b' ' + b't' -> b' t'；若是 int 则变成 32+116=148
            new_token=best_pair[0]+best_pair[1]
            merges.append(best_pair)
            new_id=len(vocab)
            vocab[new_id]=new_token

            # 错过：先写 word_freq={} 再 for word_freq.items()，
            # 字典已空，更新循环不执行，下一轮 pairs 为空直接 break。
            # 正确：用旧表生成新表，再整体替换。
            # 错过：new_word_freq={} 普通 dict 上对不存在的键做 += 会 KeyError；
            # 要用 Counter()，或 dict.get(key, 0) + freq。
            new_word_freq=Counter()

            for tokens,freq in word_freq.items():
                new_tokens=[]
                i=0
                while i<len(tokens):
                    if i<len(tokens)-1 and (tokens[i],tokens[i+1])==best_pair:
                        new_tokens.append(tokens[i]+tokens[i+1])
                        i+=2
                    else:
                        new_tokens.append(tokens[i])
                        i+=1
                # 错过：new_word_freq[key] = freq 会覆盖；
                # 多个旧序列 merge 成同一 tuple 时应累加 +=
                new_word_freq[tuple(new_tokens)]+=freq
            word_freq=new_word_freq

        return vocab,merges

