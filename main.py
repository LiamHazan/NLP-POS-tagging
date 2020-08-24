from viterbi_beam import find_absolute_tags, memm_viterbi, dict_inserts, correct_infers
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sn

if __name__ == '__main__':

    lamda = 2
    threshold = 1
    B = 5
    train_path = "train1.wtag"
    test_path = "test1.wtag"
    pickle_path = f'trained_data_1_lamda={lamda}_threshold={threshold}.pkl'
    with open(pickle_path, 'rb') as f:
      params = pickle.load(f)

    pre_trained_weights = params[0]
    feature2id = params[1]

    absolute_pairs = find_absolute_tags(train_path,100) # detects words in the train text with at least 98% recurrent tag

    with open(test_path) as f:
        words_predicted = 0
        accurate = 0
        matrix_wrong = {}
        y_actu =pd.Series([], name='Actual')
        y_pred=pd.Series([], name='Predicted')
        for line in f:
            splitted_line = line.split()
            splitted_pairs = [x.split('_') for x in splitted_line]
            splitted_pairs[0][0] = splitted_pairs[0][0].lower()
            words, tags = zip(*splitted_pairs)
            sentence = list(words)
            infer = memm_viterbi(sentence, pre_trained_weights, feature2id,B,absolute_pairs)
            print(f"real tags: {tags}")
            print(f"infer=     {infer}")
            for i in range(len(tags)):
                if tags[i]!= infer[i]:
                    dict_inserts(tags[i],matrix_wrong)
            y_actu=y_actu.append(pd.Series(tags))
            y_pred=y_pred.append(pd.Series(infer))
            accurate += correct_infers(infer,tags)
            words_predicted += len(infer)
            print("accuracy = ", accurate / words_predicted)


    most_errors = sorted(matrix_wrong.keys(), key=lambda x: matrix_wrong[x],reverse=True)[:10]

    df_confusion = pd.crosstab(y_pred,y_actu,colnames=['Actual'], rownames=['Predicted'])
    print(df_confusion)
    print(most_errors)
    df_confusion = pd.DataFrame(df_confusion,columns=most_errors)
    print(df_confusion)

    sn.heatmap(df_confusion,annot=True, fmt="d",linewidths=0.1)
    plt.xlabel("Actual")
    plt.title("Top 10 mismatched tags")
    plt.figure(figsize=(10, 3))
    plt.show()

    # print(sorted(cofusion_matrix.items(), key=lambda x: cofusion_matrix[x[0]],reverse=True)[:10])
    print("total accuracy = ",accurate/words_predicted)