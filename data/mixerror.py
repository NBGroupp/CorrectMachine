#!/usr/bin/python3
# coding: utf-8

import random
import pickle
import sys
import gzip

"""Mix errors into corpus

    Mix three types of errors(extra word, missing word, worng word in phrase)
    into sentences. Every sentence will contain just one error.

    Usage:
        python mixerror.py [corpus_path] [vocab_pkl path]
        if no lack one path, scripts will use files under ./data:
            corpus_path: ./aihanyu.gz
            vocab_path: ./vocab.pkl
"""

def is_chinese(word):
    return '\u4e00' <= word <= '\u9fff'

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


def generate_corpus(data, vocab, method):
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
            if is_chinese(sentence[random_pos]):
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


def process(corpus_path, vocab_path):
    """ Main process function

        generate three .txt files containing one error of above,
        in every new file sentence order remain same as origin corpus.

        Args:
            corpus_path: path of corpus
            vocab_path: path of vocabulary list(pkl)
                        which produced by 'data_utils.py'.
    """

    data = gzip.GzipFile(corpus_path, mode='r').read().decode().split('\n')
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)[4:]
    for i in range(3):
        print('processing error %d...' % i)
        wrong_corpus = generate_corpus(data, vocab, i)
        wrong_corpus = '\n'.join(wrong_corpus)
        wrong_corpus = gzip.compress(wrong_corpus.encode())
        with open(str(i)+'_error.gz', 'wb') as f:
            f.write(wrong_corpus)
        print(str(i)+'_error.gz'+' finished')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        corpus_path = sys.argv[1]
        vocab_path = sys.argv[2]
    else:
        corpus_path = './aihanyu.gz'
        vocab_path = './vocab.pkl'
    process(corpus_path, vocab_path)
