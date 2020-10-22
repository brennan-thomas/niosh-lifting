from processing import processing as pr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

parser = argparse.ArgumentParser(description='Print data plots')
parser.add_argument('-n', '--name', required=True, help='Name of folder to save in')
parser.add_argument('-s', '--sensors', required=True, help='The sensor channels to plot', nargs='+')
args = parser.parse_args()

pltCols = args.sensors

trim = 'min'
offset = 'max'
low_butterworth_freq = 2
hi_butterworth_freq = 12
filtered, filenames, max_size = pr.preprocess('./source', trim, offset, low_butterworth_freq, hi_butterworth_freq, do_filter=False)

#pltCols = ['RtWt_A_x', 'RtWt_A_y', 'RtWt_A_z']
count = 0
f = None
axes = None

os.makedirs('./plots/{}'.format(args.name), exist_ok=True)
currPerson = ''
currDataset = ''
print(len(filtered))
print(len(filenames))
for i in range(len(filtered)):
    print(filenames.iloc[i])
    person = filenames.iloc[i].split('_')[1]
    dataset = filenames.iloc[i].split('_')[2]
    if person != currPerson or dataset != currDataset:
        f, axes = plt.subplots(15, 3, dpi=300)
        axes = axes.flat
        f.set_figwidth(20)
        f.set_figheight(47)
        f.tight_layout()
        currPerson = person
        currDataset = dataset

    sns.lineplot(data=filtered[i].loc[:, pltCols], ax=axes[count])
    _ = axes[count].set_title(filenames.iloc[i])
    count = (count + 1) % 45

    if i == len(filenames)-1 or filenames.iloc[i+1].split('_')[1] != currPerson or filenames.iloc[i+1].split('_')[2] != currDataset:
        save_file = './plots/{}/{}_{}.pdf'.format(args.name, currPerson, currDataset)
        if os.path.exists(save_file):
            save_file = save_file[:-4] + '_none.pdf'
        f.savefig(save_file, bbox_inches='tight')
        plt.close(f)
        count = 0

