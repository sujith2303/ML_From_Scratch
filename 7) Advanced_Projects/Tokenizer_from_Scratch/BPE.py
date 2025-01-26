from collections import defaultdict
from typing import List, Optional, Tuple

class BPE:
    def __init__(self, vocab_size : int, texts : List[str], verbose : bool = False) -> None:
        self.vocab = self.create_vocab(texts)
        self.texts      = texts
        self.merges = defaultdict(str)
        self.initiate_tokenization(vocab_size, verbose)
    
    def create_vocab(self,texts : List[str]) -> List[str]:
        texts = ' '.join(texts)
        vocab = list(set(texts))
        return vocab
    
    def get_pair_freq(self):
        pair_freq = defaultdict(int)
        for word, freq in self.word_freq.items():
            for i in range(len(self.splits[word])-1):
                pair_freq[(self.splits[word][i],self.splits[word][i+1])]+=freq
        return pair_freq
    
    def create_splits(self) -> None:
        self.splits = defaultdict(list)
        for word in self.word_freq.keys():
            for i in word:
                self.splits[word].append(i)
        
    
    def get_word_freq(self, texts) -> None:
        self.word_freq = defaultdict(int)
        # texts = self.texts
        for text in texts:
            for word in text.split():
                self.word_freq[word]+=1
    
    def get_best_pair(self, pair_freq) -> Tuple[str]:
        best_pair = ''
        max_freq = 0
        for pair,freq in pair_freq.items():
            if freq>max_freq:
                max_freq = freq
                best_pair = pair
        return best_pair  

    def merge_tokens(self, token1 : str, token2: str) -> None:
        for word in self.splits.keys():
            split = self.splits[word]
            if len(split)==1:
                continue
            i = 0
            while i<len(split)-1:
                if split[i]== token1 and split[i+1] == token2:
                    split =  split[:i] + [token1+token2] + split[i+2:]
                else:
                    i+=1
            self.splits[word] = split 
    
    def initiate_tokenization(self,vocab_size : int, verbose : bool) -> None:
        texts = self.pre_tokenize(self.texts)
        self.get_word_freq(texts)
        self.create_splits()
        while len(self.vocab)<vocab_size:
            pair_freq = self.get_pair_freq()
            token1,token2 = self.get_best_pair(pair_freq)
            self.merges[(token1,token2)] = token1+token2
            self.vocab.append(token1+token2)
            if verbose:
                print(token1,token2)
                # print(f"Added {token1+token2} to vocabulary. Updated vocabulary\n",self.vocab)
            self.merge_tokens(token1, token2)
    
    def encode(self):
        pass
