from train import feature_statistics_class
from train import feature2id_class
from train import represent_input_with_features, dict_inserts, dot_product
import numpy as np
import pickle
import string
import pandas as pd



def dict_append(key, val, dictionary):
    """
        Implements the dictionary insertion for the 'find_absolute_tags' function
        :param key: dictionary key
        :param dictionary: the dictionary created in the external function

    """
    if key not in dictionary:
        dictionary[key] = [val]
    else:
        dictionary[key].append(val)


def find_absolute_tags(train,threshold):
    """
        Finds words/tags pairs with above 95% appearences as a pair
        :param train: full path of the file to read
        :param threshold: hyperparameter for the number of words recurrences in the text

            Returns a dictionary of words/tags
    """
    words_tags = {}
    chosen = {}
    with open(train) as f:
        for line in f:
            splitted_line = line.split()
            splitted_pairs = [x.split('_') for x in splitted_line]
            for pair in splitted_pairs:
                dict_append(pair[0].lower(), pair[1], words_tags)

    for word, tags in words_tags.items():
        most_common_tag = max(set(tags), key = tags.count)
        if len(tags) > threshold and tags.count(most_common_tag)/len(tags)>=0.98:
            chosen[word] = most_common_tag

    return chosen


def correct_infers(l1,l2):
    correct = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            correct += 1
    return correct


def memm_viterbi(sentence, weights, feature2id, B, absolute_pairs):
    """
        Implementation of MEMM Vitebi with Beam Search to improve runtime
        :param sentence: list of words from the test file
        :param weights: trained weights vector
        :param feature2id: instantiation of feature2id class
        :param B: hyperparameter for beam search size
        :param absolute_pairs: dictionary of frequent word/tag pairs

            Returns a list of tags for the input sentence
    """
    n = len(sentence)
    new_sen = ['*', *sentence, 'STOP']
    tags_infer = ['' for i in range(n)]  # stores the output list of the inferred tags
    n_tag_lists = ['*', '*'] + [list(feature2id.tags_set)] * n  # S_(-1),S_0,...,S_n
    tag_list = ['*'] + list(feature2id.tags_set)
    symbols = ['%','!','&','?',';',"'","`"]
    characters = list(string.punctuation)
    characters = [x for x in characters if x not in symbols]
    numbers = ['one', 'two','three','four','five','six','seven','eight','nine','ten','hundred','thousand','million','billion']
    determiners = ['a', 'an', 'every', 'no', 'the', 'these', 'this', 'those', 'some']

    # initializing a 3D list of 2D dictionaries for all kinds of tag pairs
    pi_table = []
    bp_table = []  # back pointer table
    for iter in range(n + 1):
        pi_table.append({})
        bp_table.append({})
        for i in tag_list:
            pi_table[iter][i] = {}
            bp_table[iter][i] = {}
            for j in tag_list:
                pi_table[iter][i][j] = 0
                bp_table[iter][i][j] = 'NN' # default initialization of a common tag
    pi_table[0]['*']['*'] = 1

    n_q_nominators = {}
    n_q_denominators = {}
    for k in range(n):
        for t, u in [[t0, u0] for t0 in n_tag_lists[k] for u0 in n_tag_lists[k+1]]:
            history = (t, u, new_sen[k], new_sen[k + 1], new_sen[k + 2])
            n_q_denominators[history] = 0
            for v in n_tag_lists[k+2]:
                y_history = (v, t, u, new_sen[k], new_sen[k + 1], new_sen[k + 2])
                n_q_nominators[y_history]=np.exp(dot_product(represent_input_with_features(y_history,feature2id),weights))
                n_q_denominators[history]+= n_q_nominators[y_history] #per current tag (v)

    # Algorithm
    n_qualified_tags = ['*', '*'] + [list(feature2id.tags_set)]*n
    for k in range(n):
        max_pi = 0 # stores the maximum value in the k'th 2D pi table
        B_max_bp = ['*' for i in range(B)] # back pointers for the B top pi values
        # t- pre previous tag, u- previous tag, v- current tag
        for u, v in [[u0, v0] for u0 in n_qualified_tags[k + 1] for v0 in n_qualified_tags[k + 2]]:
            max_mul = 0
            argmax = n_tag_lists[k][0]
            for t in n_qualified_tags[k]:
                history = (t, u, new_sen[k], new_sen[k + 1], new_sen[k + 2])  # new_sen[k+1] is the current word
                y_history = (v, t, u, new_sen[k], new_sen[k + 1], new_sen[k + 2])
                # q receives the index k-2 according to its entry in the sentence list input
                try:
                    mul = pi_table[k][t][u] * n_q_nominators[y_history] / n_q_denominators[history]
                    if mul > max_mul:
                        max_mul = mul
                        argmax = t
                except:
                    continue
            try:
                pi_table[k + 1][u][v] = max_mul
                bp_table[k + 1][u][v] = argmax
            except:
                continue
            if max_mul > max_pi: # updating all backpointers
                for i in range(B-1,-1,-1):
                    B_max_bp[i] = B_max_bp[i-1]
                B_max_bp[0] = v
                max_pi = max_mul
        for i in range(1,B):
            if B_max_bp[i]=='*':
                B_max_bp[i] = B_max_bp[0]
        for i in range(1,B):
            if new_sen[k + 1] in characters:
                B_max_bp[i] = new_sen[k + 1]
            if new_sen[k + 1].lower() in absolute_pairs.keys():
                B_max_bp[i] = absolute_pairs[new_sen[k + 1].lower()]
            if new_sen[k + 1].isnumeric() or new_sen[k + 1].lower() in numbers:
                B_max_bp[i] = 'CD'
            if new_sen[k + 1].lower() in determiners:
                B_max_bp[i] = 'DT'
        n_qualified_tags[k + 2] = B_max_bp  ## setting the top B tag options for the next v

    max_pi = 0  # stores the maximum value in the 2D last pi table
    for u, v in [[u0, v0] for u0 in tag_list for v0 in n_tag_lists[2]]:
        if pi_table[n][u][v] > max_pi:
            max_pi = pi_table[n][u][v]
            tags_infer[n - 2], tags_infer[n - 1] = u, v
    # bp table has entries in range 1 to n.
    # tags infer has entries in range 0 to n-1
    for k in range(n - 3, -1, -1):
        # the first tags_infer  assigning should be taken from the last entry of the bp table
        try:
            tags_infer[k] = bp_table[k + 3][tags_infer[k + 1]][tags_infer[k + 2]]
        except:
            continue
        if new_sen[k+1] in characters:
            tags_infer[k] = new_sen[k+1]
        if new_sen[k + 1].lower() in absolute_pairs.keys():
            tags_infer[k] = absolute_pairs[new_sen[k + 1].lower()]
        if new_sen[k + 1].isnumeric() or new_sen[k + 1].lower() in numbers:
            tags_infer[k] = 'CD'
        if new_sen[k + 1].lower() in determiners:
            tags_infer[k] = 'DT'

    return tags_infer

