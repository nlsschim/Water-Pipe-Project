import numpy as np
import pandas as pd

def tree_feature_importance(tree_model, X_train):
    """
    Takes in a tree model and a df of training data and prints out
    a ranking of the most important features and a bar graph of the values
    
    Parameters
    ----------
    tree_model: the trained model instance. Must have feature_importances_ and estimators_ attributes
    X_train: DataFrame that the model was training on

    Returns
    -------
    This function currently does not return any values, but that may change
    """
    importances = tree_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in tree_model.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]

    features = X_train.columns.to_list()

    # Print the feature ranking
    print("Feature ranking:")
    print()
    for f in range(X_train.shape[1]):
        #feature_name = features[indices[f]]
        print(f'{f + 1}. {features[indices[f]]}, {importances[indices[f]]}')
        print()

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
