import re
from typing import Dict, List


class Tokenizer(object):

    def __init__(self, vocab:Dict):

        self.str_to_int = vocab
        self.int_to_str = {v:k for k,v in self.str_to_int.items()}

    def encode(self, text:str):

        tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        tokens = [item if item in self.str_to_int else "<|unk|>" for item in tokens]
        ids = [self.str_to_int[s] for s in tokens]

        return ids

    def decode(self, ids:List[int]):

        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?_!"()\'])', r'\1', text)

        return text

