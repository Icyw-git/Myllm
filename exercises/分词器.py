from collections import Counter
import regex

"""
基于Unicode的划分策略
"""
import unicodedata
def get_char_category(ch:str)->str:
    cat=unicodedata.category(ch)

    if '\u4e00' <= ch <= '\u9fff': #这是CJK统一汉字的Unicode范围
        return 'CJK'

    if ch.isdigit():
        return "DIGIT"

    if ch.isalpha():
        return "ALPHA"

    if cat.startswith('P'): #标点符号的Unicode类别以'P'开头
        return "PUNCT"

    return 'OTHER'

def segment_by_unicode(text:str):
    if not text:
        return []

    buffer=[text[0]]
    segment=[]
    prev_cat=get_char_category(text[0]) #获取第一个字符的类别
    for ch in text[1:]:
        curr_cat=get_char_category(ch)
        if curr_cat==prev_cat: #如果当前字符的类别与前一个字符的类别相同，则将当前字符添加到缓冲区
            buffer.append(ch)

        else: #如果当前字符的类别与前一个字符的类别不同，则将缓冲区中的字符作为一个分段添加到结果列表中，并清空缓冲区，开始新的分段

            segment.append((''.join(buffer),prev_cat))
            buffer=[ch]
            prev_cat=curr_cat
    segment.append((''.join(buffer),prev_cat))

    tokens=[token for token,_ in segment]
    return tokens


"""
字符级tokenizer
"""

class CharacterTokenizer:
    def __str__(self):
        pass

    def encode(self,text:str):
        return [ord(ch) for ch in text] #将每个字符转换为其对应的Unicode码点（整数表示）

    def decode(self,indices):
        return ''.join([chr(i) for i in indices]) #将每个Unicode码点转换回对应的字符，并将它们连接成一个字符串


class CharTokenizer:
    def __init__(self):
        self.vocab={}
        self.inverse_vocab={}

    def encode(self,text:str):
        tokens=[]
        for ch in text:
            if ch not in self.vocab: #如果字符不在词表中，则将其添加到词表中，并为其分配一个新的索引
                idx=len(self.vocab)
                self.vocab[ch]=idx

                self.inverse_vocab[idx]=ch
            tokens.append(self.vocab[ch])
        return tokens

    def decode(self,indices):
        return ''.join(self.inverse_vocab[i] for i in indices)



"""
字节级tokenizer
"""

class ByteTokenizer:
    def __init__(self):
        self.vocab_size=256

    def encode(self,text:str):
        return list(text.encode('utf-8'))

    def decode(self,indices):
        return bytes(indices).decode('utf-8') #将字节列表转换为字节对象，并使用UTF-8解码为字符串


"""
简易BPE分词器 

"""

class BPETokenizer:

    def __init__(self,num_merges):
        self.num_merges=num_merges
        self.merges={}
        self.vocab_size=256

    def get_stats(self,tokens):
        pairs=Counter() #这里使用Counter来统计相邻字符对的频率
        for i in range(len(tokens)-1):
            pairs[(tokens[i],tokens[i+1])]+=1 #将相邻字符对作为键，频率作为值进行统计
        return pairs

    def merge_tokens(self,tokens,pair,new_token): #作用是将指定的字符对合并为一个新的token
        i=0
        new_tokens=[]
        while i<len(tokens):
            if i<len(tokens)-1 and (tokens[i],tokens[i+1])==pair:
                new_tokens.append(new_token)
                i+=2

            else:
                new_tokens.append(tokens[i])
                i+=1
        return new_tokens

    # deepseek tokenizer中使用的经典正则表达式（简化版）


TOKENIZER_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"


# 压缩率计算
def get_compression_ratio(text: str, segments):
    byte_len = len(text.encode("utf-8"))
    token_count = len(segments)
    return byte_len / token_count if token_count > 0 else 1


# Word-level Tokenizer实现
class WordTokenizer:
    def __init__(self, pattern=r"\w+|."):
        self.pattern=pattern #分词的正则表达式模式
        self.word2id={}
        self.id2word={}

    def build_vocab(self,texts):
        vocab=set()
        for text in texts:
            segments=regex.findall(self.pattern,text) #进行分词
            vocab.update(segments) #更新词表

        vocab=sorted(vocab) #对词表进行排序，确保每次构建的词表顺序一致
        self.word2id={w:i for i,w in enumerate(vocab)}
        self.id2word={i:w for w,i in self.word2id.items()}

    def decode(self,ids):
        return ''.join(self.id2word.get(i,"<UNK>") for i in ids) #解码输入的文本，若id不存在则返回<UNK>

    def encode(self,text):
        segments=regex.findall(self.pattern,text)
        return  [self.word2id.get(segment,-1) for segment in segments],segments #返回编码后的token id列表和分词结果


"""
BPE分词器简单训练
"""
# DeepSeek风格正则
DEEPSEEK_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"

def split_graphemes(token): #作用是将一个token按照Unicode的grapheme cluster进行划分，返回一个元组，包含划分后的grapheme cluster
    return tuple(regex.findall(f'\X',token))

def train_bpe(texts,num_merges=50):
    vocab=Counter() #返回一个空的Counter对象，用于统计词频
    for text in texts:
        tokens=regex.findall(DEEPSEEK_REGEX,text) #进行预处理，使用正则表达式将文本划分为token
        for token in tokens:
            chars=split_graphemes(token)+('</w>',) #将token划分为一个个字符（grapheme cluster）
            vocab[chars]+=1

    merges=[]
    for _ in range(num_merges):
        pairs=Counter()
        for word,freq in vocab.items():
            for i in range(len(word)-1): #在单词中统计相邻字符对的频率
                pairs[(word[i],word[i+1])]+=freq

        if not pairs:
            break

        best_pair=max(pairs,key=pairs.get)
        merges.append(best_pair)

        new_vocab={} #更新词表，将最频繁的字符对合并为一个新的token
        for word,freq in vocab.items():
            w=[]
            i=0
            while i<len(word):
                if i<len(word)-1 and (word[i],word[i+1])==best_pair:
                    w.append(word[i]+word[i+1])
                    i+=2
                else:
                    w.append(word[i])
                    i+=1


            new_vocab[tuple(w)]=freq #更新新的词表，将合并后的token作为键，频率作为值

        vocab=new_vocab #更新词表为新的词表，继续进行下一轮的合并操作
    return merges,vocab


class BPETokenizer:
    def __init__(self,merges):
        self.merges=merges

    def encode_word(self,token):
        word= list(split_graphemes(token)) + ["</w>"] #将token划分为grapheme cluster，并在末尾添加一个特殊的结束符号"</w>"，表示单词的结束
        for pair in self.merges:
            new_word=[]

            i=0
            while i <len(word):
                if i<len(word)-1 and (word[i],word[i+1])==pair:
                    new_word.append(word[i]+word[i+1])
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1

            word=new_word #更新word为合并后的结果，继续进行下一轮的合并操作
        return word

    def encode(self,text):
        tokens=regex.findall(DEEPSEEK_REGEX,text)
        bpe_tokens=[]
        for token in tokens:
            bpe_tokens.extend(self.encode_word(token))
        return bpe_tokens

    def decode(self,tokens):
        text=''.join(tokens).replace("</w>",'')
        return text









if __name__ == "__main__":
    test_string = "hi，很好的，terrific！🐋"
    char_count = len(test_string)
    byte_count = len(test_string.encode("utf-8"))

    # ========== 1. Unicode 划分策略 ==========
    tokens = segment_by_unicode(test_string)
    print(f"[Unicode分词] token数: {len(tokens)}, chars/token: {char_count / len(tokens):.2f}")
    print(f"[Unicode分词] tokens: {tokens}")

    # ========== 2. CharacterTokenizer（Unicode 码点级） ==========
    ct = CharacterTokenizer()
    ids = ct.encode(test_string)
    decoded = ct.decode(ids)
    assert test_string == decoded, "CharacterTokenizer 编解码不一致!"
    print(f"[CharacterTokenizer] ID数: {len(ids)}, chars/token: {char_count / len(ids):.2f}, bytes/token: {byte_count / len(ids):.2f}")

    # ========== 3. CharTokenizer（增量构建词表） ==========
    cht = CharTokenizer()
    ids1 = cht.encode(test_string)
    decoded1 = cht.decode(ids1)
    assert test_string == decoded1, "CharTokenizer 编解码不一致!"
    print(f"[CharTokenizer] 词表大小: {len(cht.vocab)}, ID数: {len(ids1)}, chars/token: {char_count / len(ids1):.2f}, bytes/token: {byte_count / len(ids1):.2f}")

    # ========== 4. ByteTokenizer（UTF-8 字节级） ==========
    bt = ByteTokenizer()
    ids2 = bt.encode(test_string)
    decoded2 = bt.decode(ids2)
    assert test_string == decoded2, "ByteTokenizer 编解码不一致!"
    print(f"[ByteTokenizer] ID数: {len(ids2)}, chars/token: {char_count / len(ids2):.2f}, bytes/token: {byte_count / len(ids2):.2f}")

    # ========== 5. 汇总对比 ==========
    print(f"\n{'Tokenizer':<22} {'ID数':<8} {'chars/token':<12} {'bytes/token':<12} {'词表大小':<10}")
    print("-" * 70)
    print(f"{'Unicode划分':<22} {len(tokens):<8} {char_count / len(tokens):<12.2f} {'N/A':<12} {'N/A':<10}")
    print(f"{'CharacterTokenizer':<22} {len(ids):<8} {char_count / len(ids):<12.2f} {byte_count / len(ids):<12.2f} {'N/A':<10}")
    print(f"{'CharTokenizer':<22} {len(ids1):<8} {char_count / len(ids1):<12.2f} {byte_count / len(ids1):<12.2f} {len(cht.vocab):<10}")
    print(f"{'ByteTokenizer':<22} {len(ids2):<8} {char_count / len(ids2):<12.2f} {byte_count / len(ids2):<12.2f} {256:<10}")
    print(f"\n原文: char数={char_count}, UTF-8字节数={byte_count}")
    print("✅ 全部测试通过！")

    string = "It's so supercalifragilisticexpialidocious!👋👋"
    print("原始字符串：", string)

    # 使用基础正则分词（基于空格和标点切分）
    basic_segments = regex.findall(r"\w+|.", string)
    print("基础正则分词结果：")
    print(basic_segments)

    # 使用deepseek风格正则
    segments = regex.findall(TOKENIZER_REGEX, string)
    print(f"deepseek风格分词结果：{segments}")

    # 构建词表
    tokenizer = WordTokenizer(pattern=TOKENIZER_REGEX)
    tokenizer.build_vocab([string])

    print("词表大小：", len(tokenizer.word2id))

    # 编码
    ids, segs = tokenizer.encode(string)
    print(f"编码token IDs：{ids}")

    # 字节序列
    byte_tokens = [b for b in string.encode("utf-8")]
    print(f"UTF-8字节序列：{byte_tokens}")

    print(f"编码segments：{segs}")

    # 解码
    decoded = tokenizer.decode(ids)
    print("解码结果：", decoded)

    # 压缩率
    ratio = get_compression_ratio(string, segs)
    print("压缩率：", ratio)

    # 高重复度语料：running/jumping/playing共享后缀-ing，runner/jumper/player共享后缀-er
    train_texts = ["running"] * 3 + ["runner"] * 2 + ["jumping"] * 3 + ["jumper"] * 2 + ["playing"] * 2 + ["player"] * 1
    merges, vocab = train_bpe(train_texts, num_merges=10)
    print("BPE合并:", merges)

    tokenizer = BPETokenizer(merges)

    # walking/walker的词干walk从未在训练中出现，但-ing/-er后缀应该能被正确切出
    for test_text in ["running", "walking", "walker", "jumper", "playground"]:
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        print(f"编码 {test_text!r}:", encoded, "  解码:", decoded)