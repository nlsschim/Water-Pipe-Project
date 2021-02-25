import time
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score


def crossval_by_year(data, classifier_fn, n_val=2, print_results=True):
    # Get all available years contained in the input data
    data_years = list(data['Process_year'].unique())

    scores = []
    t_init = time.perf_counter()
    
    for i in range(len(data_years)):
        # select window of years to use for evaluation
        val_years = data_years[i:i+n_val]

        # select relevant training and testing data, drop structural and/or invalid data
        train_df = data[~data['Process_year'].isin(val_years)] \
                     .drop(['TARGET_FID', 'Process_year', 'Break_Yr'], axis=1) \
                     .dropna(axis=0) \
                     .astype(np.float32)
        
        val_df = data[data['Process_year'].isin(val_years)] \
                     .drop(['TARGET_FID', 'Process_year', 'Break_Yr'], axis=1) \
                     .dropna(axis=0) \
                     .astype(np.float32)

        # instantiate classifier
        classifier = classifier_fn()
        # fit to training data
        classifier.fit(train_df.drop('Target', axis=1), train_df['Target'])

        # make predictions on evaluation dataset
        x_val = val_df.drop('Target', axis=1)
        y_val = val_df['Target']
        preds = classifier.predict(x_val)

        recall = recall_score(y_true=y_val, y_pred=preds)
        precision = precision_score(y_true=y_val, y_pred=preds)

        bal_acc = (recall+precision) / 2

        if (print_results):
            print(f'Cross-validation iteration {i+1} of {len(data_years)} results:')
            print(f'Balanced accuracy = {bal_acc:.4f}')
            print(f'Recall            = {recall:.4f}')
            print(f'Precision         = {precision:.4f}')
            print()

        scores.append((bal_acc, recall, precision))
    
    avgs = np.array(scores).mean(axis=0)

    if (print_results):
        print(f'Finished cross-validation (took {time.perf_counter() - t_init:.2f} seconds).')
        print(f'Avg. Balanced Acc = {avgs[0]:.4f}')
        print(f'Avg. Recall       = {avgs[1]:.4f}')
        print(f'Avg. Precision    = {avgs[2]:.4f}')

    return avgs