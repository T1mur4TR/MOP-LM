from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Tuple

import regex as re

from sentencepiece import SentencePieceTrainer, SentencePieceProcessor


class MOPiece:
    def __init__(self, directory_path: str | Path):
        directory_path = Path(directory_path)
        with open(directory_path / 'mop.data', 'r') as f:
            lines = f.readlines()
            self.min_stem = int(lines[0])
            self.prefixes = [''] * 4 + lines[1].split('|')
            self.prefix_ids = {prefix: i for i, prefix in enumerate(self.prefixes) if prefix}
            self.suffixes = [''] * 4 + lines[2].split('|')
            self.suffix_ids = {suffix: i for i, suffix in enumerate(self.suffixes) if suffix}
        with open(directory_path / 'spm.model', 'rb') as f: 
            self.spm = SentencePieceProcessor(model_proto=f.read())
        self.re = re.compile(r'([^\p{L}\p{M}\p{N}\s]+|\s)')

    def bos_id(self):
        return 0
    
    def eos_id(self):
        return 1
    
    def pad_id(self):
        return 2
    
    def unk_id(self):
        return 3
    
    def vocab_size(self):
        return len(self.prefixes), self.spm.vocab_size(), len(self.suffixes)
    
    def encode_word(self, word: str) -> Tuple[List[int], List[int], List[int]]:
        word = word.lower()
        prefix_ids = []
        suffix_ids = []
        l, r = 0, len(word)
        found = True
        while found:
            found = False
            for i in range(l + self.min_stem, r):
                if word[i:r] in self.suffix_ids:
                    suffix_ids.append(self.suffix_ids[word[i:r]])
                    r = i
                    found = True
                    break
        found = True
        while found:
            found = False
            for i in range(r - self.min_stem, l, -1):
                if word[l:i] in self.prefix_ids:
                    prefix_ids.append(self.prefix_ids[word[l:i]])
                    l = i
                    found = True
                    break
        spm_ids = self.spm.encode(word[l:r])
        return prefix_ids[::-1], spm_ids, suffix_ids[::-1]

    def encode(self, sentences: Iterable[str], morpheme_first: bool=False) -> List[List[Tuple[List[int], List[int], List[int]]]] | Tuple[List[List[List[int]]], List[List[List[int]]], List[List[List[int]]]]:
        if morpheme_first:
            prefix_tokens = []
            spm_tokens = []
            suffix_tokens = []
        else:
            tokenized_sentences = []
        for sentence in sentences:
            if morpheme_first:
                sentence_prefix_ids = []
                sentence_spm_ids = []
                sentence_suffix_ids = []
            else:
                sentence_ids = []
            for word in self.re.split(sentence):
                if word == '' or word == ' ':
                    continue
                prefix_ids, spm_ids, suffix_ids = self.encode_word(word)
                if morpheme_first:
                    sentence_prefix_ids.append(prefix_ids)
                    sentence_spm_ids.append(spm_ids)
                    sentence_suffix_ids.append(suffix_ids)
                else:
                    sentence_ids.append((prefix_ids, spm_ids, suffix_ids))
            if morpheme_first:
                prefix_tokens.append(sentence_prefix_ids)
                spm_tokens.append(sentence_spm_ids)
                suffix_tokens.append(sentence_suffix_ids)
            else:
                tokenized_sentences.append(sentence_ids)
        if morpheme_first:
            return prefix_tokens, spm_tokens, suffix_tokens
        else:
            return tokenized_sentences
    
    def decode(self, tokenized_sentences: Iterable[Iterable[Tuple[Iterable[int], Iterable[int], Iterable[int]]]] | Tuple[Iterable[Iterable[Iterable[int]]], Iterable[Iterable[Iterable[int]]], Iterable[Iterable[Iterable[int]]]], morpheme_first: bool=False) -> List[str]:
        decoded_sentences = []
        if morpheme_first:
            tokenized_sentences = zip(*tokenized_sentences)
        for sentence_ids in tokenized_sentences:
            if morpheme_first:
                sentence_ids = zip(*sentence_ids)
            words = []
            for prefix_ids, spm_ids, suffix_ids in sentence_ids:
                parts = [self.prefixes[prefix_id] for prefix_id in reversed(prefix_ids)]
                parts.append(self.spm.decode(spm_ids))
                parts.extend(self.suffixes[suffix_id] for suffix_id in suffix_ids)
                words.append(''.join(parts))
            decoded_sentences.append(' '.join(words))
        return decoded_sentences
                    

def train_mopiece(directory_path: str | Path, filepath_iterable: Iterable[str | Path], prefixes: Iterable[str], suffixes: Iterable[str], spm_vocab_size: int, spm_model_type: str='unigram', min_stem: int=3):
    prefixes_set = set(prefixes)
    suffixes_set = set(suffixes)
    reg = re.compile(r'([^\p{L}\p{M}\p{N}\s]+|\s)')
    def stems():
        for file in filepath_iterable:
            with open(file, mode='r', encoding='utf8') as file:
                for line in file.readlines():
                    for word in reg.split(line):
                        if word == '' or word == ' ':
                            continue
                        word = word.lower()
                        l, r = 0, len(word)
                        found = True
                        while found:
                            found = False
                            for i in range(l + min_stem, r):
                                if word[i:r] in suffixes_set:
                                    r = i
                                    found = True
                                    break
                        found = True
                        while found:
                            found = False
                            for i in range(r - min_stem, l, -1):
                                if word[l:i] in prefixes_set:
                                    l = i
                                    found = True
                                    break
                        yield word[l:r]
        
    spm = BytesIO()
    SentencePieceTrainer.train(sentence_iterator=stems(), vocab_size=spm_vocab_size, model_type=spm_model_type, model_writer=spm, pad_id=2, unk_id=3, bos_id=0, eos_id=1)

    directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)
    with open(directory_path / 'spm.model', 'wb') as f:
        f.write(spm.getvalue())
    with open(directory_path / 'mop.data', 'w') as f:
        f.write('\n'.join([
            str(min_stem),
            '|'.join(prefixes),
            '|'.join(suffixes)
        ]))
