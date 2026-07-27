import regex as re
import os
# 错过：用了 defaultdict 却只 from collections import Counter → NameError。
# 正确：from collections import Counter, defaultdict
from collections import Counter, defaultdict

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

        # 增量 BPE：pair_counts 全局频次；pair_to_words 倒排（哪些 word 含该 pair）
        # defaultdict(set)：key 不存在时自动空 set，可直接 .add
        pair_counts=Counter()
        pair_to_words=defaultdict(set)
        for tokens,freq in word_freq.items():
            for i in range(len(tokens)-1):
                pair_counts[(tokens[i],tokens[i+1])]+=freq
                pair_to_words[(tokens[i],tokens[i+1])].add(tokens)

        # 错过：把 merges=[] 放进 while 里，每轮清空，最后只剩 1 条 merge
        merges=[]

        while len(vocab)<vocab_size:
            if not pair_counts:
                break
            # 错过：max(pairs, key=pairs.get) —— 频次相同时只按遍历顺序，
            # 会和 reference 在某步分叉（如 index 64: (b'c',b'e') vs (b'l',b'e')）。
            # 平局时要按讲义对 pair 做字典序比较（这里用 (频次, pair)）。
            best_pair=max(pair_counts,key=lambda x: (pair_counts[x],x))
            # 这里是 bytes 拼接：b' ' + b't' -> b' t'；若是 int 则变成 32+116=148
            new_token=best_pair[0]+best_pair[1]
            merges.append(best_pair)
            new_id=len(vocab)
            vocab[new_id]=new_token

            # 错过：for tokens in pair_to_words[best_pair] 边遍历边 remove/add
            # → RuntimeError: Set changed size during iteration
            # 正确：先 list(...) 拷贝再遍历
            affected=list(pair_to_words[best_pair])
            # 错过（naive 版）：每轮扫全部 word_freq 重数 pairs + 重建全表 → ~3s，speed 测试要 <1.5s
            # 正确：只处理 affected；减旧序列全部 pair，再加新序列全部 pair（乘 freq）
            # 错过：word_freq=new_word_freq 且 new 只含 affected → 无关词全丢
            # 错过：merge 用 for i in range(len-1) 只 append 左侧、忘 i+=2 → 长度/内容错
            # 错过：pair_counts 只 -=1 而不是 -=freq；或只更新 best_pair 不更新邻接 pair
            for tokens in affected:
                freq=word_freq[tokens]
                # 1) 减旧序列贡献的全部 pair（×freq），并从倒排里摘掉旧 word
                for i in range(len(tokens)-1):
                    p=(tokens[i],tokens[i+1])
                    pair_counts[p]-=freq
                    if pair_counts[p]<=0:
                        del pair_counts[p]
                    pair_to_words[p].discard(tokens)
                # 2) while i 做不重叠 merge
                new_tokens=[]
                i=0
                while i<len(tokens):
                    if i<len(tokens)-1 and (tokens[i],tokens[i+1])==best_pair:
                        new_tokens.append(tokens[i]+tokens[i+1])
                        i+=2
                    else:
                        new_tokens.append(tokens[i])
                        i+=1
                new_tokens=tuple(new_tokens)
                # 3) 加新序列贡献的全部 pair（×freq），倒排挂上新 word
                for i in range(len(new_tokens)-1):
                    p=(new_tokens[i],new_tokens[i+1])
                    pair_counts[p]+=freq
                    pair_to_words[p].add(new_tokens)
                # 4) 更新 word_freq（可能多个旧 word 合成同一个 new）
                word_freq[new_tokens]=word_freq.get(new_tokens,0)+freq
                del word_freq[tokens]
            # best_pair 本轮已合完，倒排清空（计数应已归零并被删）
            pair_to_words[best_pair].clear()

        return vocab,merges

