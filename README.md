# niosh-lifting
Detecting lift risk levels using convolutional LSTM neural networks

## Requirements
numpy<br>
pandas<br>
scikit-learn<br>
tensorflow<br>
matplotlib<br>

## Usage
The main code runner is deepconv_kfold.py. It trains a modified DeepConvLSTM model using NIOSH lifting data and reports the model's performance. Run `python deepconv_kfold.py -h` for usage information.
