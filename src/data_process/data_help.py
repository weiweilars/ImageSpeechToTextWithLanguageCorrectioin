from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import torchaudio
import torch.nn as nn
import pdb
import os


class LetterTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        _ 28
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string)#.replace('', ' ')

    def vocab_list(self):
        return [*self.char_map]



class WordTransform():
    """Map the text to integers and dvice versa"""
    def __init__(self, pre_trained_model, max_len, do_lower_case=True):
        self.pre_trained_model = pre_trained_model
        self.max_len = max_len
        self.do_lower_case = do_lower_case
        self.tokenizer =  BertTokenizer.from_pretrained(pre_trained_model, do_lower_case=do_lower_case)
        
    def text_to_int(self, sentence):
        temp_token = []
        temp_token = [self.tokenizer.cls_token]+ self.tokenizer.tokenize(sentence)

        if len(temp_token) > self.max_len -1:
            temp_token = temp_token[:self.max_len-1]
        temp_token = temp_token + [self.tokenizer.sep_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(temp_token)
        return input_ids

    def int_to_text(self, labels):
        return self.tokenizer.convert_ids_to_tokens(labels)

    def get_vocab_len(self):
        vocab_len = len(self.tokenizer.get_vocab())
        return vocab_len

    def get_pad_token(self):
        return self.tokenizer.pad_token

    def get_tokenizer(self):
        return self.tokenizer
