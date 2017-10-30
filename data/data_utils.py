#-*- coding:utf-8 -*-
# Copyright 2017 Eric. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for data tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import gzip
import os
import re
import tarfile
import pickle
import random

from six.moves import urllib

data_path = sys.argv[1]
target_path = sys.argv[2]
vocabulary_path = sys.argv[3]

# Special vocabulary symbols - we always put them at the start.
# _PAD = "_PAD"
# _GO = "_GO"
# _EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_UNK]

# Regular expressions used to tokenize.
#_WORD_SPLIT = re.compile("(【。，！？、“‘：；）（】)")
_DIGIT_RE = re.compile(r"[\d]")
_CHAR_RE = re.compile(r"[A-Za-zＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ]+")

#PAD_ID = 0
#GO_ID = 1
#EOS_ID = 2
UNK_ID = 3

def clean_corpus(data_path, corpus_data):
    """ Clean corpus data to delete empty lines
        or line without Chinese characters.
        If the corpus is unclean, replace origin corpus_data with new data.
        Args:
            data path: origin corpus file path
            corpus_data: corpus data lists
        Returns:
            new_corpus_data: corpus data whether the corpus is clean or not.
    """
    new_corpus_data = []
    for sentence in corpus_data:
        if len(sentence) == 0:
            # empty line
            continue
        else:
            have_ch_character = ['\u4e00' <= c <= '\u9fff' for c in sentence]
            if True not in have_ch_character:
                # no chinese character
                continue
            else:
                new_corpus_data.append(sentence)
    if len(new_corpus_data) != len(corpus_data):
        # unclean, recreat corpus data
        data = '\n'.join(new_corpus_data)
        data = gzip.compress(data.encode())
        with open(data_path, 'wb') as f:
            f.write(data)
        print('clean origin corpus: {}'.format(data_path))
    else:
        print('origin corpus is clean')
    return new_corpus_data

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    #for space_separated_fragment in sentence.strip().split():
    #    words.extend(_WORD_SPLIT.split(space_separated_fragment))
    for w in sentence:
        if(w != '\n' and w != ' ' and w != '\t' and w != '\r' and w != '\f' and w != '\v'):
            words.extend(w)
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None, normalize_digits=True, normalize_char=True):
    """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
    normalize_char: Boolean; if true, all characters are replaced by a.
    """
    if not os.path.exists(vocabulary_path):
        corpus_data = gzip.GzipFile(data_path, mode='r').read().decode().split('\n')
        corpus_data = clean_corpus(data_path, corpus_data)
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}

        counter = 0
        for line in corpus_data:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for w in tokens:
                word = _DIGIT_RE.sub("0", w) if normalize_digits else w
                word = _CHAR_RE.sub("a", w) if normalize_char else w
                word = '0' if normalize_digits and (word=='０' or word=='１' or word=='２' or word=='３' or word=='４' or word=='５' or word=='６' or word=='７' or word=='８' or word=='９') else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gzip.open(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write((w + "\n").encode())
        return vocab_list
    else:
        vocab_list = gzip.GzipFile(vocabulary_path, mode='r').read().decode().split('\n')
        return vocab_list

def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = gzip.GzipFile(vocabulary_path, mode='r').read().decode().split('\n')
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True, normalize_char=True):
    """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    '''
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]
    '''
    return [vocabulary.get(w, UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True, normalize_char=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
    normalize_char: Boolean; if true, all chars are replaced by 0s.

  Return:
    tokenized_corpus: corpus that tokenized as saved into a list.
    vocab: vocabulary list
    """
    if not os.path.exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)

        corpus_data = gzip.GzipFile(data_path, mode='r').read().decode().split('\n')
        with gzip.open(target_path, mode="wb") as tokens_file:
            counter = 0
            tokenized_corpus = []
            for line in corpus_data:
                counter += 1
                if counter % 100000 == 0:
                    print("  tokenizing line %d" % counter)
                token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                        normalize_digits, normalize_char)
                to_write_data = " ".join([str(tok) for tok in token_ids])
                tokenized_corpus.append(to_write_data)
                if counter != len(corpus_data):
                    tokens_file.write((to_write_data + "\n").encode())
                else:
                    tokens_file.write(to_write_data.encode())
            return tokenized_corpus, vocab
    else:
        tokenized_corpus = gzip.GzipFile(target_path, mode='r').read().decode().split('\n')
        vocab, _ = initialize_vocabulary(vocabulary_path)
        return tokenized_corpus, vocab


def get_word(index, vocab):
    """ Get the word to use in functions of generating error to produce error.
        Guarantee word is Chinese character.
        Args:
            index: current index
            length: length of a list
        returns:
            word
    """
    length = len(vocab)
    if index < 10:
        find_range = 5
    elif index < 100:
        find_range = 50
    else:
        find_range = 100

    start = index - find_range
    if start < 0:
        start = 0
    end = index + find_range
    if end > length:
        end = length

    word = vocab[random.randrange(start, end)]
    return word

def mix_error(data, vocab, method):
    """ Generating mistakes function
        Args:
            data: corpus, list of sentences
            vocab: vocabulary that drops four special characters as start
            method: 0 - missing words method
                    1 - extra words method
                    2 - wrong words method
        returns:
            results: corpus with one wrong type,
                     a list which elements are sentences
    """
    results = []
    total_sentences = len(data)
    for i, sentence in enumerate(data):
        # one sentence
        length = len(sentence)
        if length is 0:
            continue
        print('%.2f %%' % (i/total_sentences*100), end='\r')
        while True:
            # get a Chinese character index
            random_pos = random.randrange(0, length)
            if '\u4e00' <= sentence[random_pos] <= '\u9fff':
                break
        # to processing word's index in vocabulary
        to_process_word_index = vocab.index(sentence[random_pos])

        # generate mistake
        if method is 0:
            # missing words mistake
            wrong_sentence = sentence[:random_pos] + sentence[random_pos+1:]
        else:
            # find a word in vocabulary
            error_word = get_word(to_process_word_index, vocab)
            if method is 1:
                # extra words mistake
                wrong_sentence = \
                    sentence[:random_pos] + error_word + sentence[random_pos:]
            else:
                # wrong words mistake
                wrong_sentence = sentence[:random_pos] \
                                 + error_word \
                                 + sentence[random_pos+1:]

        results.append(wrong_sentence)
    return results

def generate_corpus(data_path, target_path, vocabulary_path, vocab,
                    tokenizer=None, normalize_digits=True, normalize_char=True):

    """ Main process function

        generate three .txt files containing one error of above,
        in every new file sentence order remain same as origin corpus.

    """

    tokenized_corpus, _ = \
        data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True, normalize_char=True)

    total = len(tokenized_corpus)
    if not os.path.exists('target_train.gz'):
        with gzip.open('target_train.gz', 'wb') as f:
            f.write('\n'.join(tokenized_corpus[:int(total/3*2)]).encode())
    if not os.path.exists('target_dev.gz'):
        with gzip.open('target_dev.gz', 'wb') as f:
            f.write('\n'.join(tokenized_corpus[int(total/3*2):]).encode())

    corpus_data = gzip.GzipFile(data_path, mode='r').read().decode().split('\n')

    for i in range(3):
        print('processing error %d...' % i)
        wrong_corpus = mix_error(corpus_data, vocab, i)
        wrong_total = len(wrong_corpus)
        if wrong_total != total:
            print('wrong length of the wrong corpus')
            print(wrong_total, total)
            sys.exit()
        wrong_corpus_train = wrong_corpus[:int(total/3*2)]
        wrong_corpus_dev = wrong_corpus[int(total/3*2):]
        with gzip.open('error'+str(i)+'_train.gz', 'wb') as f:
            f.write('\n'.join(wrong_corpus_train).encode())
        with gzip.open('error'+str(i)+'_dev.gz', 'wb') as f:
            f.write('\n'.join(wrong_corpus_dev).encode())
        data_to_token_ids('error'+str(i)+'_train.gz',
                                     'error'+str(i)+'_train_token.gz', vocabulary_path)
        data_to_token_ids('error'+str(i)+'_dev.gz',
                                     'error'+str(i)+'_dev_token.gz', vocabulary_path)
        print('error'+str(i)+'.gz'+' finished')

max_vocabulary_size = 12000
vocab_list = create_vocabulary(vocabulary_path, data_path, max_vocabulary_size)
generate_corpus(data_path, target_path, vocabulary_path, vocab_list)
