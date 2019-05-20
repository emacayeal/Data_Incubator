import gensim
from gensim.models import Word2Vec

from nltk.tokenize import TweetTokenizer
# from nltk.tokenize.repp import ReppTokenizer
# from nltk.tokenize.stanford import StanfordTokenizer

from util.funcs import join

import os
import pandas as pd
import pickle as pkl
import numpy as np

import json
import yaml
import sys


def list_flatten(l):
    result = list()
    for item in l:
        if isinstance(item, (list, tuple)):
            result.extend(item)
        else:
            result.append(item)
    return result


def build_vocabulary(corpus, start_idx=1):
    corpus = list_flatten(corpus)
    return dict((word, idx) for idx, word in enumerate(set(corpus), start=start_idx))


def build_embedding(token_list, vocab, cfg, embed_type, verbose):
    assert embed_type in cfg['embed_types'],\
        'Embed type %s is not in implemented types for w2v training' % embed_type

    if cfg['use_full_review']:
        documents = []
        doc = []
        for tokens in token_list:
            break_flag = False
            if embed_type == 'word' and tokens[0] == cfg['break_str']:
                break_flag = True
            elif embed_type == 'char' and ''.join(tokens) == cfg['break_str']:
                break_flag = True
            if break_flag:
                documents.append(doc)
                doc = []
            else:
                doc.extend(tokens)
        if verbose:
            print('Using %d full reviews for w2v %s embedding' % (len(documents), embed_type))
    else:
        if embed_type == 'word':
            documents = [sent for sent in token_list if sent[0] != cfg['break_str']]
        elif embed_type == 'char':
            documents = [sent for sent in token_list if ''.join(sent) != cfg['break_str']]
        if verbose:
            print('Using %d sentences for %s w2v embedding' % (len(documents), embed_type))

    model = Word2Vec(documents, **cfg['model_args'])
    if cfg['train_embed']:
        if verbose:
            print('Training the word2vec %s model' % embed_type)
        model.train(documents, total_examples=len(documents), **cfg['train_args'])
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 2, cfg['model_args']['size']), dtype='float32')

    count = 0
    for w, i in vocab.items():
        if w not in d:
            count += 1
            emb[i, :] = np.random.uniform(-0.1, 0.1, cfg['model_args']['size'])
        else:
            emb[i, :] = weights[d[w], :]
    if verbose:
        print('embedding elements outside of vocabulary：%d' % count)
    return emb


def get_tokenizer_func(tkn_cfg):
    assert tkn_cfg['tokenizer'] in tkn_cfg['tokenizer_funcs'].keys(), \
        'Tokenizer must be in: ' + ', '.join(map(str, tkn_cfg['tokenizer_funcs'].keys()))
    if tkn_cfg['tokenizer'] == 'simple':
        pp_func = gensim.utils.simple_preprocess
    elif tkn_cfg['tokenizer'] == 'tweet':
        pp_func = TweetTokenizer(**tkn_cfg['tokenizer_funcs']['tweet']).tokenize
    pp_args = tkn_cfg['tokenizer_args'][tkn_cfg['tokenizer']]
    # def apply_func(x): return pp_func(x, **pp_args)
    return lambda x: pp_func(x, **pp_args)  # apply_func


def get_length_dict(model_input):
    text_len = [len(l) for l in model_input]
    length_dict = dict()
    length_dict['max'] = np.max(text_len)
    length_dict['min'] = np.min(text_len)
    length_dict['avg'] = np.average(text_len)
    length_dict['median'] = np.median(text_len)
    return length_dict


def run_preprocess(cfg):
    # get the configs for each part
    verbose = cfg['verbose']
    inp_cfg = cfg['input']
    tkn_cfg = cfg['token']
    w2v_cfg = cfg['w2v']
    out_cfg = cfg['out']

    # get hash of config dict
    hash_str = str(hash(json.dumps(cfg, sort_keys=True)))
    # create the new output dir
    out_dir = out_cfg['out_dir']
    if out_dir[-1] == '/':
        out_dir += 'h_%s/' % hash_str
    else:
        out_dir += '/h_%s/' % hash_str
    if os.path.exists(out_dir):
        sys.exit('An output dir with hash %s already exists' % hash_str)
    else:
        os.makedirs(out_dir)

    # input df should be of the form sentence, split, label columns
    if verbose:
        print('Beginning the preprocessing on file: ' + inp_cfg['input_file'])
    base_df = pd.read_csv(inp_cfg['input_file'], nrows=500)
    # check all the necessary columns are there
    for col in inp_cfg['check_cols'] + inp_cfg['label_cols']:
        assert col in base_df.columns,\
            'Column %s is not in the input data file %s' % (col, inp_cfg['input_file'])

    # set the tokenizer function
    tknzr_func = get_tokenizer_func(tkn_cfg)

    # tokenize and characterize all the rows
    for embed_type in tkn_cfg['use_types']:
        assert embed_type + '_col' in inp_cfg, 'Embed type %s missing from config' % embed_type
        if embed_type == 'word':
            base_df[inp_cfg['word_col']] = base_df[inp_cfg['sentence_col']].apply(tknzr_func)
        elif embed_type == 'char':
            base_df[inp_cfg['char_col']] = base_df[inp_cfg['sentence_col']].apply(lambda x: list(x))
        # optionally make all of them lower case
        if not tkn_cfg['case_sensitive']:
            base_df[inp_cfg[embed_type+'_col']] = \
                base_df[inp_cfg[embed_type+'_col']].apply(lambda x: [xx.lower() for xx in x])

    # now create and save the vocabularies
    if verbose:
        print('Getting vocabs')
    vocab_dict = {}
    for embed_type in tkn_cfg['use_types']:
        vocab = build_vocabulary(base_df[inp_cfg[embed_type+'_col']].tolist(), start_idx=1)
        vocab_dict[embed_type] = vocab
        pkl.dump(vocab, open(out_dir + embed_type + out_cfg['vocab_path'], 'wb'))
    if verbose:
        print('Length of %s vocab: %d' % (embed_type, len(vocab)))

    # create, maybe train, and save the w2v embeddings
    if verbose:
        print('Getting the w2v embeddings')
    for embed_type in tkn_cfg['use_types']:
        w2v = build_embedding(base_df[inp_cfg[embed_type+'_col']].tolist(), vocab_dict[embed_type],
                              w2v_cfg, embed_type, verbose)
        np.save(out_dir + embed_type + out_cfg['w2v_path'], w2v)

    # split the data frame after removing the break lines
    base_df = base_df[base_df[inp_cfg['sentence_col']] != w2v_cfg['break_str']]
    train_df = base_df[base_df[inp_cfg['split_col']] == inp_cfg['train_flag']]
    val_df = base_df[base_df[inp_cfg['split_col']] == inp_cfg['val_flag']]
    test_df = base_df[base_df[inp_cfg['split_col']] == inp_cfg['test_flag']]
    infer_df = base_df[base_df[inp_cfg['split_col']] == inp_cfg['infer_flag']]

    # save the length distribution info
    if verbose:
        print('Saving the embedding length info')
    for embed_type in tkn_cfg['use_types']:
        len_dict = get_length_dict(base_df[inp_cfg[embed_type+'_col']].values.tolist())
        pkl.dump(len_dict, open(out_dir + embed_type + out_cfg['len_info_path'], 'wb'))

    # all split data frames and their save strings
    df_pairs = [(train_df, 'train'), (val_df, 'val'), (test_df, 'test'), (infer_df, 'infer')]

    # prepare input
    if verbose:
        print('Saving the model input')
    for embed_type in tkn_cfg['use_types']:
        vocab = vocab_dict[embed_type]
        col = inp_cfg[embed_type+'_col']
        for df, df_str in df_pairs:
            model_input = df[col].apply(
                lambda x: [vocab.get(token, len(vocab)+1) for token in x]).tolist()
            pkl.dump(model_input, open(out_dir + embed_type + out_cfg['model_' + df_str + '_input_path'], 'wb'))

    # prepare the labels
    if verbose:
        print('Saving the labels')
    for col in inp_cfg['label_cols']:
        for df, df_str in df_pairs:
            if df_str == 'infer':
                continue
            else:
                model_output = df[col].tolist()
                pkl.dump(model_output, open(out_dir + col + out_cfg['model_' + df_str + '_output_path'], 'wb'))

    # save the config file
    with open(out_dir + out_cfg['config_file'], 'w') as outfile:
        yaml.dump(pp_cfg, outfile, default_flow_style=False)

    return


if __name__ == '__main__':
    # if running in jupyter just set config_file to the path, e.g. ./configs/preprocess_config.yaml
    config_file = sys.argv[1]  # './configs/preprocess_config.yaml'

    yaml.add_constructor('!join', join)
    with open(config_file, 'r') as yml_file:
        pp_cfg = yaml.load(yml_file)

    run_preprocess(pp_cfg)