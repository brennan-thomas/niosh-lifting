import pandas as pd
import random
import os
import pickle
import numpy as np
from numpy import genfromtxt, dstack, moveaxis, array
from processing import processing as pr
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import sys
#import keras_models as models
import heatmap as hm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain, combinations
#from keras.callbacks import EarlyStopping
import datetime
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import faulthandler
faulthandler.enable()
def powerset(iterable):
    s = list(iterable)
    tuples = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    lists = map(lambda x: list(x), tuples)
    return list(lists)

# Appends the channel postfix to the given sensors
def sensors_to_channels(sensor_list):
    channels = []
    for sensor in sensor_list:
        channels.append(sensor + "_A_x")
        channels.append(sensor + "_A_y")
        channels.append(sensor + "_A_z")
        channels.append(sensor + "_G_x")
        channels.append(sensor + "_G_y")
        channels.append(sensor + "_G_z")
    return channels


# Generates a real-valued prediction via a probability-weighted sum of classes
def classToRiskMapArrReduced(arr):
    result = []
    for el in arr:
        tot = 0
        for idx, cell in enumerate(el):
            tot += (idx + 1) * cell
        result.append(tot)
    return result

def leave_one_out_split(data_set, person):
    """
    Assumes that each person in the data (a DataFrame) has the same amount of trials.
    Chooses a range to use for testing and returns the rest as training.
    Returns a tuple of two DataFrame: (train, test)
    """
    for_training = data_set.loc[data_set['person'] != person]
    train, val = train_test_split(for_training, test_size=0.3, stratify=for_training.loc[:, 'class_label'])
    return (train, data_set.loc[data_set['person'] == person], val)

def k_fold(data_set, lo, hi):
    folds = []
    # Remove person 4 due to misaligned sensors
    data_set = data_set.loc[data_set['person'] != 4]
    for i in range(lo, hi + 1):
        if i != 4:
            folds.append(leave_one_out_split(data_set, i))
    return folds
    
def simple_split(data_set, percentage_test=0.3, num_trials=1):
    folds = []
    for i in range(num_trials):
        train, test = train_test_split(data_set, test_size=percentage_test, stratify=data_set.loc[:, 'class_label'])
        test, val = train_test_split(test, test_size=0.5, stratify=test.loc[:, 'class_label'])
        folds.append((train, test, val))
    return folds

def getclasses():
    csv = pd.read_csv('./metadata/lift_times_untrimmed.csv')
    return pd.DataFrame(csv)['class_label']

def getpeople():
    csv = pd.read_csv('./metadata/lift_times_untrimmed.csv')
    return pd.DataFrame(csv)['person']

def createFeatures(sheets, classes):
    agged = pd.DataFrame(columns=['data'])

    for i in range(len(sheets)):
        new_row = sheets[i]
        agged.loc[i] = [new_row]

    agged = agged.assign(class_label = classes)
    return agged

            
def keras_class_conversion(vec, n_classes):
    return np.array(list(map(lambda x: [1 if i == x - 1 else 0 for i in range(n_classes)], vec)))

def get_data(trim, end, allowed_sensors, preprocessing=None, simple=False, test_split = 0.25, count=1, save=False):
    # frame length of each trial. May not have enough for values > 150.
    offset = end
    # position to start from. Some trials have nothing before the lift and so break.
    # Butterworth bandpass frequency cutoffs. Look up what these mean before modifying them.
    low_butterworth_freq = 2
    hi_butterworth_freq = 12

    filtered, filenames, max_size, _ = pr.preprocess('./source', trim, offset, 
                                           low_butterworth_freq, hi_butterworth_freq, do_filter=True if preprocessing == 'filter' else False, allowed_sensors=allowed_sensors)
    classes = pr.getclasses()
    filenames = np.array(filenames)

    # pad data
    for i in range(len(filtered)):
        # pad end to length of longest trial
        df = filtered[i]
        diff = max_size - df.shape[0]
        if diff > 0:
            end_zeros = pd.DataFrame(np.zeros((diff, df.shape[1])))
            end_zeros.columns = df.columns

            filtered[i] = df.append(end_zeros)

        # pad beginning with 10 frames of zeros
        beg_zeros = pd.DataFrame(np.zeros((10, df.shape[1])))
        beg_zeros.columns = df.columns
        filtered[i] = beg_zeros.append(filtered[i])

    filtered = np.stack(filtered)

    class_labels = getclasses()
    features = createFeatures(filtered, class_labels)
    features = features.assign(person=getpeople())
    
    # Remove Person 4 because of incorrect sensors
    #features = features[features['person'] != 4]
    
    # Split the data, either for cross validation or a simple train/test split
    if not simple:
        folds = k_fold(features, 1, 10)
    else:
        folds = simple_split(features, test_split, count)

    return folds


# Train a model on one fold of data
def train_model_fold(train_data, val_data, model_name, epochs, model_num=0, batch_size=32, use_keras=False, kernel=5, test_name='test', lr=0.001, reg=0.01, dropout=0.5):
    X_train = train_data.loc[:, 'data']
    X_train = np.stack([x for x in X_train]) # extract from dataframe and stack into samples
    y_train = keras_class_conversion(train_data.loc[:, 'class_label'], 4) # one-hot embedding

    X_val = val_data.loc[:, 'data']
    X_val = np.stack([x for x in X_val])
    y_val = keras_class_conversion(val_data.loc[:, 'class_label'], 4)
    
    if use_keras:
        import keras_models as models
        from keras.callbacks import EarlyStopping, TensorBoard
    else:
        import models
        from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

    # Create the named model
    model_func = getattr(models, model_name)
    model = model_func((X_train.shape[1], X_train.shape[2]), kernel, lr=lr, reg=reg, dropout=dropout)

    # TensorBoard logging
    log_dir = 'logs/fit/{}-{}'.format(test_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
   
    # Fit the model to the fold of data, using early stopping to halt training if val loss stops decreasing
    print('Fitting model {}...'.format(model_num + 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=100, restore_best_weights=True), callback])

    return model

# Test a trained model on one fold of data
def test_model_fold(test_data, model, batch_size=32):
    X_test = test_data.loc[:, 'data']
    X_test = np.stack([x for x in X_test])
    y_test = keras_class_conversion(test_data.loc[:, 'class_label'],4) # one-hot embedding

    probs = model.predict(X_test, batch_size=batch_size)
    preds = probs.argmax(axis=-1)
    pred_mapped = preds + 1
    #pred_mapped = wholeClassToRiskMapArrReduced(preds)
    pred_prob_mapped = classToRiskMapArrReduced(probs)
    actual = test_data.loc[:, 'class_label']
    
    mat = confusion_matrix(actual, pred_mapped, labels=list(range(1,5)))

    return {'pred_mapped': pred_mapped,
            'pred_prob_mapped': pred_prob_mapped,
            'actual': actual,
            'mat': mat}

# Trains a model for every fold of data and compiles results
def train_all(folds, model_name, epochs, use_keras=False, kernel=5, test_name='test', lr=1e-5, reg=0.01, dropout=0.25, model_loaded=None):
    results = []
    models = []
    mats = []
    for idx, fold in enumerate(folds):
        train, test, val = fold

        model = train_model_fold(train, val, model_name, epochs, idx, use_keras=use_keras, kernel=kernel, test_name=test_name, lr=lr, reg=reg, dropout=dropout, model=model_loaded)
        out = test_model_fold(test, model)
        results.append(out)
        #models.append(model) #removing this bc memory seems to be an issue
        mats.append(out['mat'])

    total_mat = np.sum(x['mat'] for x in results)
    print('Total Confusion Matrix:')
    print(total_mat)

    all_probs = np.concatenate([x['pred_prob_mapped'] for x in results])
    all_pred_ints = np.concatenate([x['pred_mapped'] for x in results])
    all_actual = np.concatenate([x['actual'] for x in results])
    all_people = np.concatenate([np.full(shape=len(results[i]['actual']), 
                                         fill_value=i+1, 
                                         dtype=np.int) for i in range(len(results))])

    all_results = pd.DataFrame(
        index=range(len(all_probs)),
        data = {'Prediction': all_probs, 'Actual': all_actual,
        'Person': all_people}
    )

    youden_j = balanced_accuracy_score(all_actual, all_pred_ints, adjusted=True)
    print('Youden J Statistic:', youden_j)

    return all_results, youden_j, total_mat, models, mats


# Creates a heatmap confusion matrix and swarmplot for prediction results
def display_results(all_results, youden_j, total_mat, mats, name):
    
    hm.heatmap(total_mat, name, './trials/{}/total.png'.format(name))
    with open('./trials/{}/matrix.txt'.format(name), 'w') as f:
        f.write('Balanced Accuracy: {}\n'.format(youden_j))
        f.write(str(total_mat))
    for i, mat in enumerate(mats):
        hm.heatmap(mat, name + ' {}'.format(i + 1), './trials/{}/{}.png'.format(name, i + 1))

    f_2, ax_2 = plt.subplots(1, 1, figsize=(6, 10), dpi=300)
    sns.swarmplot(x='Actual', y='Prediction', hue='Person',
    data = all_results, ax=ax_2)
    f_2.tight_layout()
    plt.savefig('./trials/{}/swarmplot.png'.format(name))

# Trains a model using the given parameters and tests it
def run_test(args, lr=0.001, reg=0.01, dropout=0.5):
    sensors= list(set(args.keep) - set(args.remove))
    allowed_channels = sensors_to_channels(sensors)
    if args.load_data is None:
        folds, to_save = get_data(args.beginning, args.end, allowed_channels, args.preprocessing, args.simple, args.test_split, args.count)
    else:
        folds = get_saved_data(args.load_data, args.simple, args.test_split, args.count)
        to_save = None
    model_loaded = None
    if args.load_model is not None:
        model_loaded = tf.keras.models.load_model(args.load_model)
    all_results, youden_j, total_mat, models, mats = train_all(folds, args.model, args.epochs, use_keras=args.use_keras, kernel=args.kernel_size, test_name=args.name, lr=lr, reg=reg, dropout=dropout, model_loaded=model_loaded)
    display_results(all_results, youden_j, total_mat, mats, args.name)
    
    os.makedirs('models/' + args.name, exist_ok=True)
    if to_save is not None:
        pickle.dump(to_save, open('./models/{}/new_data.p'.format(args.name), 'wb'))
        for i, model in enumerate(models):
            model.save('models/{}/model{}.hdf5'.format(args.name, i))
    print('Results: ')
    print(all_results)
    print("Youden's J: ")
    print(youden_j)
    print('Total Conf Mat: ')
    print(total_mat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preprocessing', help='The type of preprocessing to do', default=None)
    parser.add_argument('-m', '--model', help='The type of model to use', default='residual_conv_oneskip_preact')
    parser.add_argument('-e', '--epochs', help='The maximum numer of epochs to train', default=300, type=int)
    parser.add_argument('-n', '--name', help='The name of this trial', default='test', type=str)
    parser.add_argument('-s', '--simple', help='Simple split instead of k-fold', action='store_true')
    parser.add_argument('-t', '--test-split', help='Percentage of data to test on if simple split', default=0.3)
    parser.add_argument('-c', '--count', help='The number of trials to run in a simple split', default=1, type=int)
    parser.add_argument('-k', '--keep', help='The sensors to keep', nargs='+', default=['Side', 'LtWt', 'RtWt', 'Back', 'UArm', 'Thig'])
    parser.add_argument('-r', '--remove', help='The sensors to remove', nargs='+', default=[])
    parser.add_argument('-u', '--use-keras', help='Use keras library', action='store_true')
    parser.add_argument('-ks', '--kernel-size', help='Kernel size for conv', default=5, type=int)
    parser.add_argument('-beg', '--beginning', help='Beginning to start sequence from (0 to start at lift start)', default='min')
    parser.add_argument('-end', '--end', help='Number of frames before ending sequence', default='max')
    parser.add_argument('-lm', '--load_model', help='Pretrained model to load', default=None)
    parser.add_argument('-ld', '--load_data', help='Leftover data to load', default=None)
    args = parser.parse_args()

    if args.beginning != 'min':
        args.beginning = int(args.beginning)
        args.end = int(args.end)
    #args.keep = ['UArm', 'Back', 'RtWt', 'Side']
    #splices = [(-100, 250), (0, 250), (0, 150)]
    #orig_name = args.name
    #for splice in splices:
    #    args.name = orig_name + str(splice)
    #    args.beginning = splice[0]
    #    args.end = splice[1]
    #    run_test(args, lr=1e-4, reg=1e-3, dropout=0.5)
    run_test(args, lr=1e-3, reg=1e-4, dropout=0.5)



