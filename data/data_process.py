# coding: utf-8

import re
import sys
import gzip
import random


_DIGIT_RE = re.compile("[\d１２３４５６７８９０]+")
_CHAR_RE = re.compile("[A-Za-zＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ]+")


def _is_chinese(c):
    return '\u4e00' <= c <= '\u9fff'


def process_corpus_data(corpus_data, normalize_char=True, normalize_digits=True):
    """ Open a gzip file and return data.
        Args:
            corpus_data: the original corpus data in list format
            normalize_digits: if true, all digits are replaced by 0
            normalize_char: if true, all chars are replaced by a
        Return:
            processed data in tuple format
    """
    p_corpus_data = []
    for one in corpus_data:
        one = _DIGIT_RE.sub('0', one) if normalize_digits else one
        one = _CHAR_RE.sub('a', one) if normalize_char else one
        p_corpus_data.append(one)
    return tuple(p_corpus_data)


def clean_corpus(data_path):
    """ Clean corpus data to delete empty lines
        or line without Chinese characters.
        If the corpus is unclean, replace origin corpus_data with new data.
        Args:
            data path: origin corpus file path
        Returns:
            new_corpus_data: clean corpus data in list format.
    """
    print('Checking origin data...')
    f = gzip.GzipFile(data_path, mode='r')
    corpus_data = f.read().decode().split('\n')
    f.close()
    new_corpus_data = []
    for sentence in corpus_data:
        if len(sentence) == 0:
            # empty line
            continue
        else:
            have_ch_character = [_is_chinese(c) for c in sentence]
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
    """ basic tokenizer: split the sentence into a list of tokens. """
    words = []
    for w in sentence:
        if(w != '\n' and w != ' ' and w != '\t' and w != '\r' and w != '\f' and w != '\v'):
            words.extend(w)
    return [w for w in words if w]


def create_vocabulary(corpus_data, max_vocabulary_size, tokenizer=None):
    """ create vocabulary from data file(contain one sentence per line)

        Args:
            corpus_data: in list format
            max_vocabulary_size: limit on the size of the created vocabulary
            tokenizer: a function to use to tokenize each data sentence
       Returns:
            a list that contains all vocabulary
    """
    print("Creating vocabulary...")
    vocab = {}

    counter = 0
    for line in corpus_data:
        counter += 1
        if counter % 100000 == 0:
            print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for word in tokens:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
        to_save_vocab_list = vocab_list[:max_vocabulary_size]
    else:
        to_save_vocab_list = vocab_list
    with open('vocab.'+str(len(to_save_vocab_list)), mode="w") as f:
        for w in to_save_vocab_list:
            f.write((w + "\n"))
    return vocab_list


def _get_word(index, vocab):
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

    while 1:
        word = vocab[random.randrange(start, end)]
        if _is_chinese(word):
            break
    return word


def create_error_corpus(corpus_data, vocab,
                        max_error_num_in_sentence=5, log=True):
    """ Create error corpus with three types of errors:
            missing characters, extra characters, wrong characters
        Each sentence is mixed with random number and type of errors
        which top to max_error_num_in_sentence argument.
        Every error character is supposed to be Chinese character.

        Args:
            corpus_data: original data in list format
            vocab: list of vocabulary
            max_error_num_in_sentence: limit of errors'number in one sentence
            log: if true, create a file to record mix process
        Return:
            error_corpus_data: in list format
            corpus_data: right corpus in list format
    """
    total_sentences = len(corpus_data)
    if log: mix_log = open('mix_log.txt', 'w')

    error_corpus = []
    no_mix_num = 0  # record no mix error sentence
    for i, sentence in enumerate(corpus_data):
        # set random error number
        chinese_character_number = sum(map(_is_chinese, sentence))
        max_error_num = min(max_error_num_in_sentence, int(chinese_character_number/10))
        error_num = random.randrange(0, max(1, max_error_num+1))
        print('Mix error: %.2f %%' % (i/total_sentences*100), end='\r')
        for _ in range(0, error_num+1):
            # set random error type
            error_type = random.randrange(0, 4)
            # one sentence
            length = len(sentence)
            while True:
                # get a Chinese character index
                random_pos = random.randrange(0, length)
                if _is_chinese(sentence[random_pos]):
                    break
            # to processing word's index in vocabulary
            to_process_word_index = vocab.index(sentence[random_pos])

            if error_type is 0:
                # missing words mistake
                if log: mix_log.write(sentence[random_pos]+' -> '+ 'None' + '\t')
                sentence = sentence[:random_pos] \
                           + sentence[random_pos+1:]
            elif error_type is 1:
                # find a word in vocabulary
                error_word = _get_word(to_process_word_index, vocab)
                # extra words mistake
                if log: mix_log.write(sentence[random_pos] + ' -> ' \
                                      + sentence[random_pos] + error_word + '\t')
                sentence = sentence[:random_pos] \
                                 + error_word + sentence[random_pos:]
            else:
                # wrong words mistake
                error_word = _get_word(to_process_word_index, vocab)
                if log: mix_log.write(sentence[random_pos] \
                                      + ' -> ' + error_word + '\t')
                sentence = sentence[:random_pos] \
                                 + error_word + sentence[random_pos+1:]
        if log: mix_log.write('\n')
        if sentence == corpus_data[i]:
            no_mix_num += 1
        error_corpus.append(sentence)
    print('Mix error: Done.          ')
    if log:
        mix_log.seek(0)
        mix_log.write('Correct: {} / {}\n'.format(no_mix_num, len(corpus_data)))
        mix_log.close()
    return error_corpus, corpus_data

def generate_data(data_path, max_vocabulary_size):
    corpus_data = clean_corpus(data_path)
    corpus_data = process_corpus_data(corpus_data)
    vocab_l = create_vocabulary(corpus_data, max_vocabulary_size)
    oerror_corpus, oright_corpus = create_error_corpus(corpus_data, vocab_l,
                                                       max_error_num_in_sentence=3)
    # insert space into sentence
    error_corpus = []
    for one in oerror_corpus:
        n_one = ' '.join(one)
        error_corpus.append(n_one)
    right_corpus = []
    for one in oright_corpus:
        n_one = ' '.join(one)
        right_corpus.append(n_one)

    data_len = len(error_corpus)
    # split into dev(0.2), val(0.2), train(0.6)
    two_split_index = int(data_len/10*2)
    four_split_index = int(data_len/10*4)
    dev_error_corpus = error_corpus[0:two_split_index]
    dev_right_corpus = right_corpus[0:two_split_index]
    val_error_corpus = error_corpus[two_split_index: four_split_index]
    val_right_corpus = right_corpus[two_split_index: four_split_index]
    train_error_corpus = error_corpus[four_split_index:]
    train_right_corpus = right_corpus[four_split_index:]
    with open('error.dev.'+str(len(dev_error_corpus)), 'w') as f:
        f.write('\n'.join(dev_error_corpus))
    with open('error.val.'+str(len(val_error_corpus)), 'w') as f:
        f.write('\n'.join(val_error_corpus))
    with open('error.train.'+str(len(train_error_corpus)), 'w') as f:
        f.write('\n'.join(train_error_corpus))
    with open('right.dev.'+str(len(dev_right_corpus)), 'w') as f:
        f.write('\n'.join(dev_right_corpus))
    with open('right.val.'+str(len(val_right_corpus)), 'w') as f:
        f.write('\n'.join(val_right_corpus))
    with open('right.train.'+str(len(train_right_corpus)), 'w') as f:
        f.write('\n'.join(train_right_corpus))


if __name__ == '__main__':
    data_path = sys.argv[1]
    max_vocabulary_size = 7000
    generate_data(data_path, max_vocabulary_size)
