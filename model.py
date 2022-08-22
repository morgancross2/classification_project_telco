# standard ds imports
import numpy as np
import pandas as pd

# for model evaluation
import sklearn.metrics as met
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def binary_decisiontree_data(X_train, y_train):
    '''
    ONLY FOR A BINARY TARGET. 
    (Built on titanic data where target was survived (0 or 1))
    a
    This function takes in:
        X_train = train dataset minus target series as DataFrame 
        y_train = target variable column as a series
        
    Returns a DataFrame running decision tree models from 2 to 7 depth with confusion matrix data.
    '''
    trees = {}
    z = 1
    for i in range(2,8):
        tree = DecisionTreeClassifier(max_depth=i)
        tree.fit(X_train, y_train)

        cm = met.confusion_matrix(y_train, tree.predict(X_train))
        TP = cm[1,1]
        FP = cm[0,1]
        TN = cm[0,0]
        FN = cm[1,0]

        acc = round(((TP+TN)/(TP+FP+FN+TN))*100,2)
        TPR = round((TP/(TP+FN))*100,2)
        TNR = round(((TN)/(FP+TN))*100,2)
        FPR = round((FP / (FP + TN))*100,2)
        FNR = round(((FN)/(TP+FN))*100,2)
        percision = round((TP/(TP+FP))*100,2)
        f1 = round((met.f1_score(y_train, tree.predict(X_train)))*100,2)
        sp = TP + FN
        sn = FP + TN

        model_name = 'tree '+str(z)
        trees[model_name] = {'max_depth': i,
                         'accuracy' : acc.astype(str)+'%', 
                         'recall_TPR': TPR.astype(str)+'%', 
                         'specificity_TNR': TNR.astype(str)+'%', 
                         'FPR': FPR.astype(str)+'%', 
                         'FNR': FNR.astype(str)+'%', 
                         'percision': percision.astype(str)+'%',
                         'f1': f1.astype(str)+'%',
                         'support_pos': sp,
                         'support_neg': sn}
        z+=1
    return pd.DataFrame(trees).T


def binary_randomforest_data(X_train, y_train):
    '''
    ONLY FOR A BINARY TARGET. 
    (Built on titanic data where target was survived (0 or 1))
    
    This function takes in:
        X_train = train dataset minus target series as DataFrame
        y_train = target variable column as a series
        
    Returns a DataFrame running random forest models with confusion matrix data
    from 2 to 20 depth (increment by 2) and 1 to 30 (increment by 5) min sample leaf.
    '''
    forests = {}
    z = 1
    for i in range(2,20,2):
        for x in range(1,31,5):
            forest = RandomForestClassifier(max_depth=i, min_samples_leaf=x, random_state=123)
            forest.fit(X_train, y_train)
    
            cm = met.confusion_matrix(y_train, forest.predict(X_train))
            TP = cm[1,1]
            FP = cm[0,1]
            TN = cm[0,0]
            FN = cm[1,0]
    
            acc = round(((TP+TN)/(TP+FP+FN+TN))*100,2)
            TPR = round((TP/(TP+FN))*100,2)
            TNR = round(((TN)/(FP+TN))*100,2)
            FPR = round((FP / (FP + TN))*100,2)
            FNR = round(((FN)/(TP+FN))*100,2)
            percision = round((TP/(TP+FP))*100,2)
            f1 = round((met.f1_score(y_train, forest.predict(X_train)))*100,2)
            sp = TP + FN
            sn = FP + TN
            
            model_name = 'forest ' + str(z)
            forests[model_name] = {'max_depth' : i,
                         'min_samples_leaf' : x,
                         'accuracy' : acc.astype(str)+'%', 
                         'recall_TPR': TPR.astype(str)+'%', 
                         'specificity_TNR': TNR.astype(str)+'%', 
                         'FPR': FPR.astype(str)+'%', 
                         'FNR': FNR.astype(str)+'%', 
                         'percision': percision.astype(str)+'%',
                         'f1': f1.astype(str)+'%',
                         'support_pos': sp,
                         'support_neg': sn}
            z += 1
    return pd.DataFrame(forests).T


def binary_KNN_data(X_train, y_train, weight):
    '''
    ONLY FOR A BINARY TARGET. 
    (Built on titanic data where target was survived (0 or 1))
    
    This function takes in:
        X_train = train dataset minus target series as DataFrame
        y_train = target variable column as a series
        weights = uniform or distance as a string
        
    Returns a DataFrame running K Nearest Neighbor models with confusion matrix data
    from 1 to 20 neighbors.
    '''
    models = {}
    z = 1
    for i in range(1,21):
        model = KNeighborsClassifier(n_neighbors=i, weights=weight)
        model.fit(X_train, y_train)
        
        cm = met.confusion_matrix(y_train, model.predict(X_train))
        TP = cm[1,1]
        FP = cm[0,1]
        TN = cm[0,0]
        FN = cm[1,0]
        
        acc = round(((TP+TN)/(TP+FP+FN+TN))*100,2)
        TPR = round((TP/(TP+FN))*100,2)
        TNR = round(((TN)/(FP+TN))*100,2)
        FPR = round((FP / (FP + TN))*100,2)
        FNR = round(((FN)/(TP+FN))*100,2)
        percision = round((TP/(TP+FP))*100,2)
        f1 = round((met.f1_score(y_train, model.predict(X_train)))*100,2)
        sp = TP + FN
        sn = FP + TN
        
        model_name = 'knn ' + str(z)
        models[model_name] = {'neighbors': i,
                        'accuracy' : acc.astype(str)+'%', 
                        'recall_tpr': TPR.astype(str)+'%', 
                        'specificity_tnr': TNR.astype(str)+'%', 
                        'fpr': FPR.astype(str)+'%', 
                        'fnr': FNR.astype(str)+'%', 
                        'percision': percision.astype(str)+'%',
                        'f1': f1.astype(str)+'%',
                        'support_pos': sp,
                        'support_neg': sn}
        z += 1
    return pd.DataFrame(models).T


def binary_logisticregression_data(X_train, y_train, penalty='l2'):
    '''
    ONLY FOR A BINARY TARGET. 
    (Built on titanic data where target was survived (0 or 1))
    
    This function takes in:
        X_train = train dataset minus target series as DataFrame
        y_train = target variable column as a series
        penalty = default to l2, can apply l1, l2, none, or elasticnet
        
    Returns two DataFrames (models and odds):
        Models df - Creates Logistic Regression models with confusion matrix data
                    from .01 to 1000 C (increment logrithmically)
    
        Odds df - Displays odds of X_train features
                a. If the coefficient (odds) is 1 or close to 1, this means odds of 
                    being in class '1' (positive class) is same or close to being in class 
                    '0' (negative class). This means the feature with this coefficient is 
                    not a big driver for the target variable in this particular model.
                b. If the coefficient value is < 1 , that implies that increase in 
                    value of that feature will decrease the odds that target variable is 
                    in positive class.
    '''
    models = {}
    odds = {}
    z = 1
    span = [.01, .1, 1, 10, 100, 1000]
    for i in range(0,6):
        model = LogisticRegression(C=span[i])
        model.fit(X_train, y_train)
        
        cm = met.confusion_matrix(y_train, model.predict(X_train))
        TP = cm[1,1]
        FP = cm[0,1]
        TN = cm[0,0]
        FN = cm[1,0]
        
        acc = round(((TP+TN)/(TP+FP+FN+TN))*100,2)
        TPR = round((TP/(TP+FN))*100,2)
        TNR = round(((TN)/(FP+TN))*100,2)
        FPR = round((FP / (FP + TN))*100,2)
        FNR = round(((FN)/(TP+FN))*100,2)
        percision = round((TP/(TP+FP))*100,2)
        f1 = round((met.f1_score(y_train, model.predict(X_train)))*100,2)
        sp = TP + FN
        sn = FP + TN
        
        model_name = 'logit ' + str(z)
        models[model_name] = {'C': span[i],
                        'accuracy' : acc.astype(str)+'%', 
                        'recall_tpr': TPR.astype(str)+'%', 
                        'specificity_tnr': TNR.astype(str)+'%', 
                        'fpr': FPR.astype(str)+'%', 
                        'fnr': FNR.astype(str)+'%', 
                        'percision': percision.astype(str)+'%',
                        'f1': f1.astype(str)+'%',
                        'support_pos': sp,
                        'support_neg': sn}
        odds[model_name] = model.coef_[0]
        z += 1
    models = pd.DataFrame(models).T
    odds = np.exp(pd.DataFrame(odds, index=X_train.columns)).T
    odds['C'] = models['C']
        
    return models, odds


def setup(train, val, test):
    ''' This function takes in the train, validate, and test datasets and returns the 
    adjusted X_train, y_train, X_val, y_val, X_test, y_test for the desired columns in
    modeling for the telco classification project.
    '''
    # Drop columns with high chi-square p-value with churn
    drop_cols = ['partner', 
                 'dependents', 
                 'phone_service', 
                 'multiple_lines',
                 'Bank transfer (automatic)',
                 'Credit card (automatic)',
                 'Electronic check',
                 'Mailed check',
                 # 'contract_type',
                 'total_charges',
                 # 'Fiber optic',
                 # 'DSL',
                 # 'monthly_charges',
                 # 'paperless_billing',
                 'gender'
                 # 'senior_citizen'
                 # 'tenure'
                 # 'extras'
    ]

    train = train.drop(columns=drop_cols)
    val = val.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)

    # Create model variables for train, validate, and test modeling
    X_train = train.drop(columns=['churn', 'customer_id'])
    y_train = train.churn
    X_val = val.drop(columns=['churn', 'customer_id'])
    y_val = val.churn
    X_test = test.drop(columns=['churn', 'customer_id'])
    y_test = test.churn
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_logit(X_train, y_train, X_val, y_val):
    ''' This function takes in X_train, y_train, X_validate, y_validate and returns
    the logistic regression model's accuracy and recall results for hyperparameters:
     - C = 1.0
     - frequency = 0.3
     - random_state = 123
    '''
    # Make the logistic regression model
    logit3 = LogisticRegression(random_state=123)
    # Fit it to the train data
    logit3.fit(X_train, y_train)

    # Evaluate it with a 0.3 threshold 
    y_pred_proba_train = logit3.predict_proba(X_train)
    y_pred_proba_train = pd.DataFrame(y_pred_proba_train, columns = ['stayed', 'churned'])
    y_pred_train = (y_pred_proba_train.churned > 0.3).astype(int)
    logit_accuracy = met.accuracy_score(y_train, y_pred_train)
    logit_recall = met.recall_score(y_train, y_pred_train)

    y_pred_proba_val = logit3.predict_proba(X_val)
    y_pred_proba_val = pd.DataFrame(y_pred_proba_val, columns = ['stayed', 'churned'])
    y_pred_val = (y_pred_proba_val.churned > 0.3).astype(int)
    val_acc = met.accuracy_score(y_val, y_pred_val)
    val_recall = met.recall_score(y_val, y_pred_val)

    # Put the results into a dataframe
    logit_results = pd.DataFrame({'logit_results' : {'accuracy_train': logit_accuracy,
                                                     'accuracy_validate' : val_acc,
                                                     'recall_train': logit_recall,
                                                     'recall_validate': val_recall,}
                         })
    return logit_results


def make_forest(X_train, y_train, X_val, y_val):
    ''' This function takes in X_train, y_train, X_validate, y_validate and returns
    the random forest model's accuracy and recall results for hyperparameters:
     - max_depth = 7
     - min_samples_leaf = 3
     - random_state = 123
    '''
    # Make the Random Forest model
    forest23 = RandomForestClassifier(max_depth=7, min_samples_leaf=3, random_state=123)
    # Fit it to the train data
    forest23.fit(X_train, y_train)

    # Evaluate and put the results in a dataframe
    forest_results = pd.DataFrame({'forest_results' : {'accuracy_train': forest23.score(X_train, y_train),
                                                       'accuracy_validate' : forest23.score(X_val, y_val),
                                                       'recall_train': met.recall_score(y_train, forest23.predict(X_train)),
                                                       'recall_validate': met.recall_score(y_val, forest23.predict(X_val))}
                 })
    return forest_results


def make_knn(X_train, y_train, X_val, y_val):
    ''' This function takes in X_train, y_train, X_validate, y_validate and returns
    the K-nearest neighbor model's accuracy and recall results for hyperparameters:
     - n-neighbors = 5
     - weights = uniform
    '''
    # Make the KNN model
    knn5 = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    # Fit it to the train data
    knn5.fit(X_train, y_train)

    # Evaluate and put the results in a dataframe
    knn_results = pd.DataFrame({'knn_results' : {'accuracy_train': knn5.score(X_train, y_train),
                                                 'accuracy_validate' : knn5.score(X_val, y_val),
                                                 'recall_train': met.recall_score(y_train, knn5.predict(X_train)),
                                                 'recall_validate': met.recall_score(y_val, knn5.predict(X_val))}
                 })
    return knn_results


def run_best_test(X_train, y_train, X_val, y_val, X_test, y_test, test):
    ''' This function takes in X_train, y_train, X_validate, y_validate, X_test, y_test, and test
    and returns 3 items:
    
    1. The logistic regression model's accuracy and recall results for hyperparameters:
     - C = 1.0
     - frequency = 0.3
     - random_state = 123
     
    2. The model's coefficient/weight on each feature in a dataframe. 
    
    3. The model's predictions on the test dataset. This is also saved into a local csv file.
    '''
    # Make the logistic regression model
    logit3 = LogisticRegression()
    # Fit it on the train data
    logit3.fit(X_train, y_train)

    # Evaluate it for each dataset: train, validate, test
    y_pred_proba_train = logit3.predict_proba(X_train)
    y_pred_proba_train = pd.DataFrame(y_pred_proba_train, columns = ['stayed', 'churned'])
    y_pred_train = (y_pred_proba_train.churned > 0.3).astype(int)
    logit_accuracy = met.accuracy_score(y_train, y_pred_train)
    logit_recall = met.recall_score(y_train, y_pred_train)

    y_pred_proba_val = logit3.predict_proba(X_val)
    y_pred_proba_val = pd.DataFrame(y_pred_proba_val, columns = ['stayed', 'churned'])
    y_pred_val = (y_pred_proba_val.churned > 0.3).astype(int)
    val_acc = met.accuracy_score(y_val, y_pred_val)
    val_recall = met.recall_score(y_val, y_pred_val)

    y_pred_proba_test = logit3.predict_proba(X_test)
    y_pred_proba_test = pd.DataFrame(y_pred_proba_test, columns = ['stayed', 'churned'])
    y_pred_test = (y_pred_proba_test.churned > 0.3).astype(int)
    test_acc = met.accuracy_score(y_test, y_pred_test)
    test_recall = met.recall_score(y_test, y_pred_test)

    # Put the results into a dataframe
    best = pd.DataFrame({'Accuracy' : {
                             'train': logit_accuracy,
                             'validate' : val_acc,
                             'test' : test_acc},
                         'Recall' : {
                             'train': logit_recall,
                             'validate': val_recall,
                             'test': test_recall}
                        })
    
    odds = np.exp(pd.DataFrame(logit3.coef_[0], index=X_train.columns, columns=['weight_of_feature']))
    
    # Create the columns
    predict_stay = np.array(y_pred_proba_test.stayed)
    predict_churn = np.array(y_pred_proba_test.churned)
    churn_or_not = np.where(y_pred_test == 1, 'Churn', 'Stay')
    customers = np.array(test.customer_id)

    # Put it into a dataframe
    predictions = pd.DataFrame([customers, predict_stay, predict_churn, churn_or_not]).T
    # Clean up the column names
    predictions.columns = ['customer', 'probability_of_staying', 'probability_of_churn', 'prediction']
    # Send it to a local csv file
    predictions.to_csv('predictions.csv',index=False)
    # Show us the dataframe
    
    return best, odds, predictions