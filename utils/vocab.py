import pickle
from tqdm import tqdm
from collections import Counter
from utils.tokenizer import Tokenizer
import multiprocessing

class Vocab(object):
    def __init__(self, min_freq=10):
        self.vocab = Counter()
        self.min_freq = min_freq 
        self.word2id = None
        self.id2word = None
        self.vocab_size = None
    
    def load(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            tmp = pickle.load(f)
        self.vocab = tmp['vocab']
        self.min_freq = tmp['min_freq']
        self.word2id = tmp['word2id']
        self.id2word = tmp['id2word']
        self.vocab_size = tmp['vocab_size']

    def save(self, vocab_path):
        vocab = {
            "vocab" : self.vocab,
            "min_freq" : self.min_freq,
            "word2id" : self.word2id,
            "id2word" : self.id2word,
            "vocab_size" : self.vocab_size
        }
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)

    def create(self, file_name, lang):
        nlp = Tokenizer(lang)

        print("-----------loading-----------")
        with open(file_name, encoding='utf-8') as f:
            lines = f.readlines()
        f = open(file_name+'.token', 'w', encoding='utf-8')
        for line in tqdm(lines):
            token = nlp.tokenizer(line)
            self.vocab.update(token)
            l = '\t'.join(token)
            f.write(l + '\n')
        
        tmp = self.vocab.most_common()
        tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        tokens += [i[0] for i in tmp if i[1] > self.min_freq]
        self.word2id = {word:idx for idx, word in enumerate(tokens)}
        self.id2word = {idx:word for word, idx in self.word2id.items()}
        self.vocab_size = len(self.word2id)


if __name__ == "__main__":
    de = 'de_core_news_sm'
    en = 'en_core_web_sm'

    de_train = '../dataset2/train.de'
    en_train = '../dataset2/train.en'

    de_valid = '../dataset2/val.de'  # 注意：应该是 val.de 不是 valid.de
    en_valid = '../dataset2/val.en'  # 注意：应该是 val.en 不是 valid.en

    de_vocab = Vocab()
    en_vocab = Vocab()

    # 只处理训练集和验证集
    de_vocab.create(de_train, de)
    print(f"德语训练集词汇表大小: {de_vocab.vocab_size}")
    de_vocab.create(de_valid, de)
    print(f"最终德语词汇表大小: {de_vocab.vocab_size}")

    en_vocab.create(en_train, en)
    print(f"英语训练集词汇表大小: {en_vocab.vocab_size}")
    en_vocab.create(en_valid, en)
    print(f"最终英语词汇表大小: {en_vocab.vocab_size}")

    de_vocab.save('../dataset2/de_vocab.pkl')
    en_vocab.save('../dataset2/en_vocab.pkl')