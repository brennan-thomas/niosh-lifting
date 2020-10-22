import vis
from vis.visualization import visualize_saliency, visualize_cam
import keras
from vis.utils import utils
import matplotlib.pyplot as plt
import deepconv_kfold
import models
import numpy as np
import os
import argparse
import pandas as pd

cols = ['Side_A_x', 'Side_A_y', 'Side_A_z', 'Side_G_x', 'Side_G_y', 
        'Side_G_z', 'LtWt_A_x', 'LtWt_A_y', 'LtWt_A_z', 'LtWt_G_x', 
        'LtWt_G_y', 'LtWt_G_z', 'RtWt_A_x', 'RtWt_A_y', 'RtWt_A_z', 
        'RtWt_G_x', 'RtWt_G_y', 'RtWt_G_z', 'Back_A_x', 'Back_A_y', 
        'Back_A_z', 'Back_G_x', 'Back_G_y', 'Back_G_z', 'UArm_A_x', 
        'UArm_A_y', 'UArm_A_z', 'UArm_G_x', 'UArm_G_y', 'UArm_G_z', 
        'Thig_A_x', 'Thig_A_y', 'Thig_A_z', 'Thig_G_x', 'Thig_G_y', 
        'Thig_G_z']
#cols = np.flip(cols)

DO_CHANNELS=True

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--preprocessing', help='The type of preprocessing to do', default=None)
parser.add_argument('-m', '--model', help='The type of model to use', default='residual_conv_oneskip_preact')
parser.add_argument('-e', '--epochs', help='The number of epochs to train', default=50, type=int)
parser.add_argument('-n', '--name', help='The name of this trial', default='test', type=str)
parser.add_argument('-f', '--fold', help='The fold to perform saliency on', default=None)
parser.add_argument('-l', '--layer', help='The layer to perform saliency on', default=7, type=int)
parser.add_argument('-c', '--channels', help='Use channels in saliency', action='store_true')
parser.add_argument('-s', '--simple', help='Simple split instead of k-fold', action='store_true')
parser.add_argument('-t', '--test-split', help='Percentage of data to test on if simple split', default=0.25, type=float)
parser.add_argument('-r', '--runs', help='The number of trials to run in a simple split', default=1, type=int)
parser.add_argument('-i', '--risk-index', help='0=Low, 1=Medium, 2=High', type=int, default=None)
args = parser.parse_args()


def init():
    if os.path.exists('./saliency_data/{}'.format(args.name)):
        models = []
        folds = []
        for i in range(len(os.listdir('./saliency_data/{}/models'.format(args.name)))):
            models.append(keras.models.load_model('./saliency_data/{}/models/model{}.hdf5'.format(args.name, i)))
            folds.append((pd.read_pickle('./saliency_data/{}/data/train{}.p'.format(args.name, i)), pd.read_pickle('./saliency_data/{}/data/test{}.p'.format(args.name, i)), pd.read_pickle('./saliency_data/{}/data/val{}.p'.format(args.name, i))))
    else:
        models = []
        folds = deepconv_kfold.get_data('min', 'max', cols, args.preprocessing, args.simple, args.test_split, args.runs)
        if args.fold is not None:
            folds = [folds[args.fold]]
        for i, fold in enumerate(folds):
            train, test, val = fold
            model = deepconv_kfold.train_model_fold(train, val, args.model, args.epochs, i, use_keras=True)
            models.append(model)
            os.makedirs('./saliency_data/{}/models/'.format(args.name), exist_ok=True)
            os.makedirs('./saliency_data/{}/data/'.format(args.name), exist_ok=True)
            model.save('./saliency_data/{}/models/model{}.hdf5'.format(args.name, i))
            test.to_pickle('./saliency_data/{}/data/test{}.p'.format(args.name, i))
            train.to_pickle('./saliency_data/{}/data/train{}.p'.format(args.name, i))
            val.to_pickle('./saliency_data/{}/data/val{}.p'.format(args.name, i))
    return models, folds

def plot_saliency_total(gradients, risk, model, do_channels=False):
    grads = gradients

    labels = ['Low Risk Saliency', 'Medium Risk Saliency', 'High Risk Saliency', 'No Lift Saliency']
    if risk is not None:
        risk = labels[risk[0]]
        plt.title(risk)
    else:
        plt.title('Overall Saliency')

    plt.xlabel('Frame')

    if do_channels:
        grads = np.rot90(grads)
        #grads = grads[:, 1]
        #grads = np.expand_dims(grads, axis=0)
        #grads = np.repeat(grads, 36, axis=0)
        plt.yticks(ticks=np.linspace(-0.975, 0.975, 36), labels=cols)
        plt.ylabel('Sensor Channel')
        plt.imshow(grads, cmap="jet", aspect="auto", extent=(0, 615, -1, 1))
    else:
        grads = np.expand_dims(grads, axis=0)
        grads = np.repeat(grads, 50, axis=0)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        plt.imshow(grads, cmap="jet")
    #upperarm_y_a_data = X_test[i, :, 25]

    save_dir = './saliency/{}/{}/'.format(args.name, model)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + str(risk) + '.png')
    plt.clf()

def saliency_total(model, X_test, risk, layer, do_channels=False):
    #model.layers[-1].activation = keras.activations.linear
    #model = utils.apply_modifications(model)

    model.summary()

    plt.figure(figsize=(15, 10))
    print('SHAPE')
    print(X_test.shape)   
    grads = visualize_saliency(model, layer, filter_indices=risk, seed_input=X_test, keepdims=do_channels)
    
    return grads

def test_model(model, X_test, y_test):
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    t = model.predict(X_test)

    classes = []
    for j in t:
        classes += [[1 if i == max(j) else 0 for i in j]]
    classes = np.array(classes)

    confusion = [[0 for _ in range(4)] for y in range(4)]

    for i in range(len(y_test)):
        idx_act = np.where(y_test[i] == 1)[0][0]
        idx_pred = np.where(classes[i] == 1)[0][0]
        confusion[idx_act][idx_pred] += 1

    confusion = np.array(confusion)
    print('Confusion:')
    print(confusion)

    return accuracy, confusion

def format_data(data):
    X = data.loc[:, 'data']
    X = np.stack([x for x in X])
    y = deepconv_kfold.keras_class_conversion(data.loc[:, 'class_label'], 4)

    return X, y

if __name__ == '__main__':
    models, folds = init()

    risk = args.risk_index
    if risk is not None:
        risk=[risk]

    for i, (model, fold) in enumerate(zip(models, folds)):
        train, test, val = fold
        X_test, y_test = format_data(test)
        acc, conf = test_model(model, X_test, y_test) 
        grads = saliency_total(model, X_test, risk, args.layer, args.channels)
        plot_saliency_total(grads, risk, i, args.channels)
