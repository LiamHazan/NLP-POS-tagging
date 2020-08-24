import numpy as np
import pickle

dict_names = ['count_dict_100','prefixes_dict','suffixes_dict','count_dict_103','count_dict_104','count_dict_105',
              'count_dict_106', 'count_dict_107','count_dict_108','count_capital_dict','count_all_caps_dict',
              'count_numbers_dict']
# contains a dictionary field of words and tags
class feature_statistics_class():
    """
        Fills the dictionaries with respective features and empirical counts, by reading the train file
        Creates a set of tags and a list of histories according to the text
    """
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated
        self.histories = []
        self.tags_set = set([])
        # Initialize all features dictionaries
        self.all_dicts = {}
        for dict_name in dict_names:
            self.all_dicts[dict_name] = {}

    def get_count_dicts(self, file_path):
        """
            Extract out of text all kinds of desired words/tags combinations features
            :param file_path: full path of the file to read
                stores in dictionaries all features with number of recurrences
        """

        with open(file_path) as f:
            for line in f:
                splitted_line = ['*_*', '*_*'] + line.split() + ['STOP_STOP']
                splitted_pairs = [x.split('_') for x in splitted_line]
                splitted_pairs[2][0] = splitted_pairs[2][0].lower()
                for index, pair in enumerate(splitted_pairs):
                    if pair[0] == '*' or pair[0] == 'STOP':
                        continue
                    cur_word, cur_tag, pre_prev_tag = pair[0], pair[1], splitted_pairs[index - 2][1]
                    prev_tag, prev_word = splitted_pairs[index - 1][1], splitted_pairs[index - 1][0]
                    next_word, next_tag = splitted_pairs[index + 1][0], splitted_pairs[index + 1][1]
                    self.tags_set.add(cur_tag)

                    # word/tag features
                    self.histories.append((cur_tag, pre_prev_tag, prev_tag, prev_word, cur_word, next_word))
                    dict_inserts((cur_word, cur_tag), self.all_dicts['count_dict_100'])

                    # spelling features
                    for i in range(1, 5):
                        dict_inserts((cur_word[:i],cur_tag), self.all_dicts['prefixes_dict'])
                        dict_inserts((cur_word[-i:],cur_tag), self.all_dicts['suffixes_dict'])

                    # contextual features
                    dict_inserts((pre_prev_tag, prev_tag, cur_tag), self.all_dicts['count_dict_103'])
                    dict_inserts((prev_tag, cur_tag), self.all_dicts['count_dict_104'])
                    dict_inserts(cur_tag, self.all_dicts['count_dict_105'])
                    dict_inserts((prev_word, cur_tag), self.all_dicts['count_dict_106'])
                    dict_inserts((next_word, cur_tag), self.all_dicts['count_dict_107'])
                    dict_inserts((cur_word,pre_prev_tag, prev_tag, cur_tag), self.all_dicts['count_dict_108'])
                    if cur_word[0].isupper():
                        dict_inserts(cur_tag, self.all_dicts['count_capital_dict'])
                    if cur_word.isupper():
                        dict_inserts(cur_tag, self.all_dicts['count_all_caps_dict'])
                    if cur_word.isnumeric():
                        dict_inserts(cur_tag, self.all_dicts['count_numbers_dict'])


def dict_inserts(key, dictionary):
    """
        Implements the insertion to dictionaries and counts the recurrences
    """
    if key not in dictionary:
        dictionary[key] = 1
    else:
        dictionary[key] += 1


class feature2id_class():
    """
        Maintains only features which surpass the threshold, and assign each with an index
        Creates a vectors list of representing histories with features indexes
    """

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        self.tags_set = set(feature_statistics.tags_set)
        self.total_features_appearences = 0
        self.histories = feature_statistics.histories
        self.histories_represented = []
        self.empirical_counts = 0
        self.all_dicts = {}

        for dict_name in dict_names:
            self.all_dicts[dict_name] = {}

        self.n_total_features = 0  # Total number of features accumulated

    def get_features_an_id(self):
        """
            Assings each feature with an index in an incrementing order
            Counts total amount of features
        """
        for dict_name, dict in self.feature_statistics.all_dicts.items():
            for feature, count in dict.items():
                if count >= self.threshold:
                    self.all_dicts[dict_name][feature] = (self.n_total_features, count) # (index,count)
                    self.n_total_features += 1
                    self.total_features_appearences += count

        print("total_features=",self.n_total_features)

    def represent_histories(self):
        """
            Represent the histories using the 'represent_input_with_features' function
        """
        total_histories = len(self.histories)
        self.histories_represented = [ [] for i in range(total_histories) ]
        n = self.n_total_features
        # self.history_features_count = np.zeros(n)
        for i in range(total_histories):
            features = represent_input_with_features(self.histories[i],self)
            self.empirical_counts += indeces_to_n_vector(features, n)
            # every entry has a history features list and list of all possible history features lists
            self.histories_represented[i].append(features)
            self.histories_represented[i].append([])
            for tag in self.tags_set:
                y_history = (tag,*self.histories[i][1:6])
                self.histories_represented[i][1].append(represent_input_with_features(y_history,self))

def calc_objective_per_iter(v, feature2id, lamda):
    """
        Calculate max entropy likelihood for an iterative optimization method
        :param v: weights vector in iteration i
        :param feature2id: instantiation of feature2id class
        :param lamda: hyperparameter for regularization

            The function returns the Max Entropy log-linear likelihood (objective) and the objective gradient
    """
    total_features = feature2id.n_total_features
    total_histories = len(feature2id.histories_represented)
    expected_counts = np.zeros(total_features)
    normalization_term = 0

    for i in range(total_histories):
        denominator = 0
        add2_expected_counts = 0
        for features in feature2id.histories_represented[i][1]:
            nominator = np.exp(dot_product(features, v),dtype=np.float64)
            add2_expected_counts += indeces_to_n_vector(features, total_features) * nominator
            denominator += nominator
        expected_counts += add2_expected_counts / denominator
        normalization_term += np.log(denominator,dtype=np.float64)

    empirical_counts = feature2id.empirical_counts
    regularization = 0.5 * lamda * ((np.linalg.norm(v)) ** 2)
    regularization_grad = v * lamda

    linear_term = np.dot(empirical_counts, v)
    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad
    print("end of calc_objective call")

    return (-1) * likelihood, (-1) * grad


def represent_input_with_features(history, feature2id):
    """
        Extract  indexes feature vector per a given history
        :param history: tuple{ctag, pptag, ptag, pword, word, nword}
        :param feature2id: instantiation of feature2id class
            Return a list with all features indexes that are relevant to the given history
    """
    ctag = history[0]
    pptag = history[1]
    ptag = history[2]
    pword = history[3]
    word = history[4]
    nword = history[5]
    features = []

    if ctag in feature2id.all_dicts['count_dict_105']:
        features.append(feature2id.all_dicts['count_dict_105'][ctag][0]) # adds the relevant index of the pair

        # for i in range(1, 4):
        if (word[:1], ctag) in feature2id.all_dicts['prefixes_dict']:
            features.append(feature2id.all_dicts['prefixes_dict'][(word[:1], ctag)][0])
            if (word, ctag) in feature2id.all_dicts['count_dict_100']:
                features.append(feature2id.all_dicts['count_dict_100'][(word, ctag)][0])
            if (word[:2], ctag) in feature2id.all_dicts['prefixes_dict']:
                features.append(feature2id.all_dicts['prefixes_dict'][(word[:2], ctag)][0])
                if (word[:3], ctag) in feature2id.all_dicts['prefixes_dict']:
                    features.append(feature2id.all_dicts['prefixes_dict'][(word[:3], ctag)][0])
                    if (word[:4], ctag) in feature2id.all_dicts['prefixes_dict']:
                        features.append(feature2id.all_dicts['prefixes_dict'][(word[:4], ctag)][0])
        if (word[-1:], ctag) in feature2id.all_dicts['prefixes_dict']:
            features.append(feature2id.all_dicts['prefixes_dict'][(word[-1:], ctag)][0])
            if (word[-2:], ctag) in feature2id.all_dicts['prefixes_dict']:
                features.append(feature2id.all_dicts['prefixes_dict'][(word[-2:], ctag)][0])
                if (word[-3:], ctag) in feature2id.all_dicts['prefixes_dict']:
                    features.append(feature2id.all_dicts['prefixes_dict'][(word[-3:], ctag)][0])
                    if (word[-4:], ctag) in feature2id.all_dicts['prefixes_dict']:
                        features.append(feature2id.all_dicts['prefixes_dict'][(word[-4:], ctag)][0])

        if (ptag, ctag) in feature2id.all_dicts['count_dict_104']:
            features.append(feature2id.all_dicts['count_dict_104'][(ptag, ctag)][0])

            if (pptag, ptag, ctag) in feature2id.all_dicts['count_dict_103']:
                features.append(feature2id.all_dicts['count_dict_103'][(pptag, ptag, ctag)][0])

                if (word, pptag, ptag, ctag) in feature2id.all_dicts['count_dict_108']:
                    features.append(feature2id.all_dicts['count_dict_108'][(word, pptag, ptag, ctag)][0])

        if (pword, ctag) in feature2id.all_dicts['count_dict_106']:
            features.append(feature2id.all_dicts['count_dict_106'][(pword, ctag)][0])

        if (nword, ctag) in feature2id.all_dicts['count_dict_107']:
            features.append(feature2id.all_dicts['count_dict_107'][(nword, ctag)][0])

        if word[0].isupper() and ctag in feature2id.all_dicts['count_capital_dict']:
            features.append(feature2id.all_dicts['count_capital_dict'][ctag][0])

        if word.isupper() and ctag in feature2id.all_dicts['count_all_caps_dict']:
            features.append(feature2id.all_dicts['count_all_caps_dict'][ctag][0])

        if word.isnumeric() and ctag in feature2id.all_dicts['count_numbers_dict']:
            features.append(feature2id.all_dicts['count_numbers_dict'][ctag][0])

    return features

def indeces_to_n_vector(features, n):
    """
        Creates a full binary features vector, using a short features indexes vector
        :param features: features indexes vector
        :param n: total amount of features
            Return a binary vector corresponds to the relevant features
    """
    res = np.zeros(n)
    for f in features:
        res[f] = 1
    return res

def dot_product(features, v):
    """
        Implement a shortened dot product of weights vectors and features
        :param features: features indexes vector
        :param v: weights vector
            Return a dot product scalar
    """
    res = 0
    for i in features:
        res += v[i]
    return res

def init_w_0 (feature2id):
    """
        Initialize a weights vector corresponding to the empirical counts of each feature
        :param feature2id: instantiation of feature2id class
            Return a weights vector
    """
    w_0 = np.zeros(feature2id.n_total_features, dtype=np.float64)
    total_counts = feature2id.total_features_appearences

    for dict_name, dict in feature2id.all_dicts.items():
        for index, count in dict.values():
            w_0[index] += count / total_counts

    return w_0



if __name__ == '__main__':

    from scipy.optimize import fmin_l_bfgs_b

    train_path = "train1.wtag"

    # parameters for reading a pickle before optimizing again
    lamda = 2
    threshold = 1
    pickle_path = 'trained_data_1_lamda=2_threshold=1.pkl'
    with open(pickle_path, 'rb') as f:
        params = pickle.load(f)
    w_0 = params[0]

    statistics = feature_statistics_class()
    statistics.get_count_dicts(train_path)

    feature2id = feature2id_class(statistics, threshold)
    feature2id.get_features_an_id()
    feature2id.represent_histories()

    # w_0 = init_w_0(feature2id)
    args = (feature2id,lamda) #arguments passed to fmin_l_bfgs_b
    optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=4, iprint=99)
    weights = optimal_params[0]

    # IMPORTANT - we expect to recieve weights in 'pickle' format, don't use any other format!!
    feature2id.histories_represented = []
    pickle_path = f'trained_data_1_lamda={lamda}_threshold={threshold}.pkl'  # i identifies which dataset this is trained on
    trained_params = [weights,feature2id]
    with open(pickle_path, 'wb') as f:
        pickle.dump(trained_params, f)
