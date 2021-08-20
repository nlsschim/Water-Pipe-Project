import time
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def crossval_by_year(data, classifier_fn, params_dict, n_val=2, threshold=0.5, print_results=True, normalize_data=False):
    '''
    data:          the dataframe with all of the pipe break data
    classifier_fn: the sklearn classifier that you want to use
    params_dict:   a dictionary containing all of the parameters you want to use for your classifier

    Example:

    df = pd.read_csv('final_data.csv')
    params = {
      'n_estimators': 150,
      'max_depth': 5
    }
    avg_acc = crossval_by_year(df, ExtraTreesClassifier, params)

    returns a numpy array with [avg balanced accuracy, avg recall, avg precision]
    '''
    # Get all available years contained in the input data
    data_years = list(data['Process_year'].unique())

    assert(n_val < len(data_years))

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
        classifier.set_params(**params_dict)
        # fit to training data
        x_train = train_df.drop('Target', axis=1)
        y_train = train_df['Target']

        if normalize_data:
          scaler = StandardScaler()
          scaler.fit(x_train)
          x_train = scaler.transform(x_train)

        classifier.fit(x_train, y_train)

        # make predictions on evaluation dataset
        x_val = val_df.drop('Target', axis=1)
        y_val = val_df['Target']
        if normalize_data:
          x_val = scaler.transform(x_val)
        
        pred_prob = classifier.predict_proba(x_val)
        preds = (pred_prob[:,1] >= threshold).astype(bool) 

        recall = recall_score(y_true=y_val, y_pred=preds)
        precision = precision_score(y_true=y_val, y_pred=preds)

        bal_acc = balanced_accuracy_score(y_true=y_val, y_pred=preds)

        train_preds = classifier.predict(x_train)

        train_recall = recall_score(y_train, train_preds)
        train_precision = precision_score(y_train, train_preds)

        train_bal_acc = balanced_accuracy_score(y_train, train_preds)

        if (print_results):
            print(f'Cross-validation iteration {i+1} of {len(data_years)} results:')
            print(f'Balanced accuracy = {bal_acc:.4f}')
            print(f'Recall            = {recall:.4f}')
            print(f'Precision         = {precision:.4f}')
            print()
            print(f'Training set balanced accuracy = {train_bal_acc}')
            print(f'Train Recall            = {train_recall:.4f}')
            print(f'Train Precision         = {train_precision:.4f}')
            print()

        scores.append((bal_acc, recall, precision))

    avgs = np.array(scores).mean(axis=0)

    if (print_results):
        print(f'Finished cross-validation (took {time.perf_counter() - t_init:.2f} seconds).')
        print(f'Avg. Balanced Acc = {avgs[0]:.4f}')
        print(f'Avg. Recall       = {avgs[1]:.4f}')
        print(f'Avg. Precision    = {avgs[2]:.4f}')

    return avgs

# make sure the two datasets have the same columns
def crossval_between_cities(train_df, val_df, classifier_fn, params_dict, n_val=2, threshold=0.5, print_results=True, normalize_data=False):
    # Get all available years contained in the input data
    data_years = list(val_df['Process_year'].unique())
    print(data_years)

    scores = []
    t_init = time.perf_counter()

    # use all years for training data
    train_df = train_df.drop(['TARGET_FID', 'Process_year', 'Break_Yr'], axis=1) \
                    .dropna(axis=0) \
                    .astype(np.float32)

    # instantiate classifier
    classifier = classifier_fn()
    classifier.set_params(**params_dict)
    # fit to training data
    x_train = train_df.drop('Target', axis=1)
    y_train = train_df['Target']

    if normalize_data:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)

    classifier.fit(x_train, y_train)
    
    for i in range(len(data_years)):
        # select window of years to use for evaluation
        val_years = data_years[i:i+n_val]

        # select relevant training and testing data, drop structural and/or invalid data
        year_val = val_df[val_df['Process_year'].isin(val_years)] \
                     .drop(['TARGET_FID', 'Process_year', 'Break_Yr'], axis=1) \
                     .dropna(axis=0) \
                     .astype(np.float32)

        # make predictions on evaluation dataset
        x_val = year_val.drop('Target', axis=1)
        y_val = year_val['Target']
        if normalize_data:
          x_val = scaler.transform(x_val)
        
        pred_prob = classifier.predict_proba(x_val)
        preds = (pred_prob[:,1] >= threshold).astype(bool) 

        recall = recall_score(y_true=y_val, y_pred=preds)
        precision = precision_score(y_true=y_val, y_pred=preds)

        bal_acc = balanced_accuracy_score(y_true=y_val, y_pred=preds)

        if (print_results):
            print(f'Cross-validation iteration {i+1} of {len(data_years)} results:')
            print(f'Balanced accuracy = {bal_acc:.4f}')
            print(f'Recall            = {recall:.4f}')
            print(f'Precision         = {precision:.4f}')
            print()

        scores.append((bal_acc, recall, precision))
    
    avgs = np.array(scores).mean(axis=0)


    train_preds = classifier.predict(x_train)

    train_recall = recall_score(y_train, train_preds)
    train_precision = precision_score(y_train, train_preds)

    train_bal_acc = balanced_accuracy_score(y_train, train_preds)


    if (print_results):
        print(f'Finished cross-validation (took {time.perf_counter() - t_init:.2f} seconds).')
        print(f'Avg. Balanced Acc = {avgs[0]:.4f}')
        print(f'Avg. Recall       = {avgs[1]:.4f}')
        print(f'Avg. Precision    = {avgs[2]:.4f}')
        print()
        print(f'Training set balanced accuracy = {train_bal_acc}')
        print(f'Train Recall            = {train_recall:.4f}')
        print(f'Train Precision         = {train_precision:.4f}')

    return avgs
