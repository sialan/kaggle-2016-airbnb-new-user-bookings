import random
import psycopg2
from sklearn.externals import joblib
from datetime import datetime
import os
from random import sample, seed
from collections import defaultdict, Counter
import pandas as pd
from sklearn.metrics import accuracy_score
import csv
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def reservoir_sampling(N_population, n_samples):
    results = []
    for i, item in enumerate(N_population):
        if i < n_samples:
            results.append(item)
        else:
            k = random.randint(0, i)
            if k < n_samples:
                results[k] = item

    return results

def collect_classifier_predictions(data_dict, **kwargs):
    """Given a data dictionary  containing 'train_data' and 'test_data' (as pandas DFs) and classifiers (kwargs),
    This runs the classifier and outputs the predictions of each classifier as a dictionary.
    Input:
        data_dict: the data dictionary containing all the train/test data/targets
        kwargs: sequence of classifiers (e.g. RF = RandomForest(), lin_svc = LinearSVC()...
    Output:
        dictionary of predictions where the key is the classifier label given in kwargs and the value is a list of predictions"""
    
    pred_dict = {}
    for classifier in kwargs.keys():
        
        # Fit a model on all the data and features
        kwargs[classifier].fit(data_dict['train_data'], data_dict['train_targets'])

        # Make predictions on dev data
        pred_dict[classifier] = kwargs[classifier].predict(data_dict['test_data'])
    
    # Return the dev performance score.
    return pred_dict

def ensemble_voting(predictions_dict):
    '''Takes in the predictions dictionary output from collect_classifier_predictions and returns pred with most votes'''
    
    #Instantiate an object to hold the combined scores from each classifier
    scores = defaultdict(list)
    
    #Run through each classifier and get voting predictions
    for classifier in predictions_dict.keys():
        
        for i, prediction in enumerate(predictions_dict[classifier]):
            scores[i].append(prediction)
    
    final_preds = []
    for i in sorted(scores):
        final_preds.append(Counter(scores[i]).most_common(1)[0][0])
    
    return pd.Series(final_preds, dtype = 'object')

def save_model(model, model_name, save_dir = 'models/', record_keeping_file = 'models/record_keeping.csv'):
    '''In order to manage our models, we need to keep track of where and when they came from. Each time
    this function is called, it serializes 'model' to a file called 'model_name'.pickle in a newly created folder located
    in 'save_dir' and writes a log of the event as a new line in 'record_keeping_file'

    Input:
        model: model object created using scikit-learn
        model_name: the name you'd like to give the model; this name will be the name of the new_folder
            created to house the model
        save_dir: the filepath of the directory in which to save the model (defaults to 'models/')
        record_keeping_file: the filepath of the file which keeps a record (name and date) of all created models
            (defaults to 'models/record_keeping.csv')
    '''
    
    #Create the new folder to house the model
    new_folder = save_dir + model_name
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    #Serialize the model
    complete_fp = new_folder + '/' + model_name + '.pickle'
    with open(complete_fp, 'wb') as f:
        joblib.dump(model, complete_fp)
    
    #Write the event to the record-keeping file (format = model_name, serialized_filepath, current_time)
    with open(record_keeping_file, 'a') as f:
        f.write(model_name + ',' + complete_fp + ',' + str(datetime.now()) + '\n')
    
    return

def run_classifier(classifier, data_dict):
    """Given a classifier and a data dictionary containing 'train_data' and 'test_data' (as pandas DFs),
    This runs the classifier and outputs the accuracy of the classifier on the test data."""
    
    # Fit a model on all the data and features
    classifier.fit(data_dict['train_data'], data_dict['train_targets'])

    # Make predictions on dev data
    dev_predictions = classifier.predict(data_dict['test_data'])
    
    # Return the dev performance score.
    return accuracy_score(data_dict['test_targets'], dev_predictions)

def randomly_sample_pitchers(cursor, num_pitchers = 5, min_pitch_count = 600, seed_num = None):
    '''Takes a random sample of pitchers from the db represented by "cursor" and returns a Pandas DF with
    the specified number ofpitchers who have thrown at least "min_pitch_count" pitches
    Input:
        cursor: DB handle
        num_pitchers: The number of pitchers whose data you want returned
        min_pitch_count: Minimum number of pitches a pitcher must have thrown in order to be considered in the 
            random sampling
        seed_num: If you want to be able to replicated the results, set a seed
    Output: Pandas DF containing pitch data for the randomly sampled pitchers'''
    
    cur = cursor
    
    #Get all pitchers meeting the min pitches criterion
    get_pitchers_query = '''SELECT pitcher, count(*)
                        FROM all_pitch_data
                        GROUP BY pitcher
                        HAVING count(*) >= %d''' % min_pitch_count
    cur.execute(get_pitchers_query)
    
    #Get all the pitcher ids and sample from them
    if seed_num is not None:
        seed(seed_num)
    
    pitcher_ids = [pitch_id for (pitch_id, counter) in cur.fetchall()]
    pitcher_id_sample = sample(pitcher_ids, num_pitchers)
    
    #Grab all pitch data from these pitchers
    get_pitches_query = '''SELECT *
                            FROM all_pitch_data
                            where pitcher in (%s)''' % str(pitcher_id_sample).strip('[]')
    cur.execute(get_pitches_query)
    
    #Create Pandas DF and return it
    rows = cur.fetchall()
    header = [colnames[0] for colnames in cur.description]
    pitcher_df = pd.DataFrame(rows)
    pitcher_df.columns = header
    
    return(pitcher_df)

def convert_same_pitches(pitch_df):
    '''According to http://www.beyondtheboxscore.com/2011/3/31/2068855/pitch-fx-primer, "FT" and "SI" are
    the same pitch and "KC" and "CU" are the same pitch.  This function takes a Pandas pitch DF and converts
    these pitch types to simply "FT" and "CU"
    Input:
        pitch_df: Pandas DF containing pitch_data with field "pitch_type"
    Output: Pandas DF with corrected pitch types'''
    
    pitch_df.pitch_type[pitch_df.pitch_type == 'FT'] = 'SI'
    pitch_df.pitch_type[pitch_df.pitch_type == 'KC'] = 'CU'
    
    return pitch_df

def load_model(model_name, record_keeping_file = 'models/record_keeping.csv'):
    '''This function takes in the name of a model and searches record_keeping_file
    for that name. It then tries to load and return the model (the last instance listed in record_keeping_file
    if there are more than one).  It requires that you run this function from the root directory of our project.
    
    Inputs:
        model_name: string of the model's name as it appears in record_keeping_file
        record_keeping_file: the filename of the file that holds the record keeping info for saved models
            (defaults to 'models/record_keeping.csv')
            
    Returns: De-serialized model matching the name of model_name    
    '''
    
    #open the record keeping file and get the filepath of the last instance where model_name occurs
    with open(record_keeping_file, 'rb') as f:
        reader = csv.reader(f, delimiter = ',')
        for row in reader:
            print row[0]
            if row[0] == model_name:
                model_fp = row[1]
    
    #Return the model
    return joblib.load(model_fp)

def order_correlations(prediction_dict):
    '''Function to determine how correlated predictions are among classifiers.  Classifiers with high accuracy
    that are less correlated are better candidates for ensembling. This seems like inefficient code...
    Note: this only works for pitch types "Fastball" and "Not_Fastball"
    Input:
        predictions_dict: dictionary of predictions where key = classifier name and value = predictions
    Output: returns nothing but prints a table listing correlation among predictions in decreasing order'''
    
    #Get a list of all the classifiers and initialize the list to store everything
    classifiers = prediction_dict.keys()
    corr_list = []
    
    #Loop through all combinations of classifiers and calculate the correlation coefficient
    for i in range(len(classifiers)):
        classifier1 = classifiers[i]
        for j in range((i + 1), len(classifiers)):
            classifier2 = classifiers[j]
            pred1 = np.where(prediction_dict[classifier1] == 'Fastball', 1, 0)
            pred2 = np.where(prediction_dict[classifier2] == 'Fastball', 1, 0)
            correlation = pearsonr(pred1, pred2)
            corr_list.append([classifier1, classifier2, correlation])
            
    #order the correlations from least to greatest
    corr_list = sorted(corr_list, key = itemgetter(2))
    
    #Print everything nicely
    print 'classifier1\tclassifier2\tcorrelation'
    for item in corr_list:
        print '%s\t%s\t%f'% (item[0], item[1], item[2][0])
        
    return

def randomly_sample_pitchers2(cursor, num_pitchers = 5, min_pitch_count = 600, min_date = '2015-01-01', seed_num = None):
    '''Takes a random sample of pitchers from the db represented by "cursor" and returns a Pandas DF with
    the specified number ofpitchers who have thrown at least "min_pitch_count" pitches
    Input:
        cursor: DB handle
        num_pitchers: The number of pitchers whose data you want returned
        min_pitch_count: Minimum number of pitches a pitcher must have thrown in order to be considered in the 
            random sampling
        seed_num: If you want to be able to replicated the results, set a seed
    Output: Pandas DF containing pitch data for the randomly sampled pitchers'''
    
    cur = cursor
    
    #Get all pitchers meeting the min pitches criterion
    get_pitchers_query = '''SELECT pitcher, 
                                COUNT(*) as tot_pitch_count, 
                                MAX(date) as maximum_date
                        FROM all_pitch_data_reclass
                        GROUP BY pitcher
                        HAVING count(*) >= %d AND 
                                MAX(date) > '%s'
                        ORDER BY pitcher''' % (min_pitch_count, min_date)
    cur.execute(get_pitchers_query)
    
    #Get all the pitcher ids and sample from them
    if seed_num is not None:
        seed(seed_num)
    
    rows = cur.fetchall()
    header = [colnames[0] for colnames in cur.description]
    pitcher_df = pd.DataFrame(rows)
    pitcher_df.columns = header
    
    pitcher_id_sample = sample(pitcher_df['pitcher'].values, num_pitchers)
    
    pitcher_df = pitcher_df[pitcher_df['pitcher'].isin(pitcher_id_sample)]
    
    return pitcher_df

def naive_accuracy(data_dict):
    '''Calculate the accuracy of just guessing the most common pitch, using the test data from the data dictionary'''

    biggest_count = data_dict['test_targets'].value_counts()[0]
    all_counts = data_dict['test_targets'].value_counts().sum()
    return round(float(biggest_count) / all_counts, 3)

def subset_data(modeling_dict, cols_of_interest):
    '''Subset the training and testing data sets in the modeling_dict to only include the list of columns "cols_of_interest"'''
    new_dict = modeling_dict.copy()
    new_dict['train_data'] = new_dict['train_data'][cols_of_interest]
    new_dict['test_data'] = new_dict['test_data'][cols_of_interest]
    return new_dict

def run_all_classifiers(data_dict):
    '''Takes in a modeling dictionary and runs the following classifiers:
    - Random Forest
    - Gradient Boosted Machine
    - Logistic Regression
    - Linear Support Vector Machine
    Returns a dictionary with these four trained models'''
    
    #Initialize a dictionary to hold all the classifiers
    classifier_dict = {}
    classifier_dict['rf'] = (RandomForestClassifier(max_depth=3,
                                                   min_samples_leaf = 7,
                                                   min_samples_split = 6,
                                                   n_estimators = 350)
                             .fit(data_dict['train_data'], data_dict['train_targets']))
    classifier_dict['gbm'] = (GradientBoostingClassifier(max_depth=3,
                                                         loss = 'deviance',
                                                         max_features = 'auto')
                              .fit(data_dict['train_data'], data_dict['train_targets']))
    classifier_dict['log_reg'] = (LogisticRegression(C = 0.1,
                                                penalty = 'l1')
                             .fit(data_dict['train_data'], data_dict['train_targets']))
    classifier_dict['lin_svc'] = (LinearSVC(C = 0.1,
                                            penalty = 'l1',
                                            dual = False)
                                  .fit(data_dict['train_data'], data_dict['train_targets']))
    
    return classifier_dict

def collect_classifier_predictions2(data_dict, classifier_dict):
    """Given a data dictionary  containing 'train_data' and 'test_data' (as pandas DFs) and classifiers (kwargs),
    This runs the classifier and outputs the predictions of each classifier as a dictionary.
    Input:
        data_dict: the data dictionary containing all the train/test data/targets
        classifier_dict: dictionary of trained classifiers
    Output:
        dictionary of predictions where the key is the classifier label given in kwargs and the value is a list of predictions"""
    
    pred_dict = {}
    for classifier in classifier_dict.keys():

        # Make predictions on dev data
        pred_dict[classifier] = classifier_dict[classifier].predict(data_dict['test_data'])
    
    # Return the dev performance score.
    return pred_dict

def choose_best_ensemble(pred_dict, modeling_dict):
    '''Taking in a dictionary of predictions from different models and the modeling_dict, determines which combinations
    of classifiers work best on the test data'''

    #initialize best accuracy
    best_accuracy = 0
    
    #Try each of the classifiers individually
    for classifier in pred_dict:
        new_acc = accuracy_score(modeling_dict['test_targets'], pred_dict[classifier])
        
        if new_acc > best_accuracy:
            best_accuracy = new_acc
            classifier_combo = classifier
    
    # Using at least three classifiers, try all different modeling combinations
    for i in range(3, len(pred_dict.keys()) + 1):
        
        for combo in combinations(pred_dict.keys(), i):
            
            #reformulate the pred dictionary based on the current combo
            new_dict = dict((k, pred_dict[k]) for k in combo)
            
            #Ensemble vote
            new_preds = ensemble_voting(new_dict)
            
            #Get accuracy and compare to current best
            new_acc = accuracy_score(modeling_dict['test_targets'], new_preds)
            if new_acc > best_accuracy:
                best_accuracy = new_acc
                classifier_combo = combo
    
    return {'best_acc' : best_accuracy,
            'classifier_combination' : classifier_combo}