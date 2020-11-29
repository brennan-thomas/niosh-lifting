# niosh-lifting
Detecting lift risk levels using convolutional LSTM neural networks

# Requirements
numpy
pandas
scikit-learn
tensorflow
matplotlib

The main code runner is deepconv_kfold.py. It trains a modified DeepConvLSTM model using NIOSH lifting data and reports the model's performance. Run `python deepconv_kfold.py -h` for usage information.
