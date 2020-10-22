from processing import processing as pr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

#parser = argparse.ArgumentParser(description='Print data plots')
#parser.add_argument('-n', '--name', required=True, help='Name of folder to save in')
#parser.add_argument('-s', '--sensors', required=True, help='The sensor channels to plot', nargs='+')
#args = parser.parse_args()

#pltCols = args.sensors

trim = -10#'min'
offset = 110#'max'
low_butterworth_freq = 2
hi_butterworth_freq = 12
filtered, filenames, max_size, starts = pr.preprocess('./source', trim, offset, low_butterworth_freq, hi_butterworth_freq, do_filter=False)


start = starts[0]
data = filtered[0]
name = filenames.iloc[0]#person = filenames.iloc[0].split('_')[1]
#os.makedirs('./plots/{}'.format(args.name), exist_ok=True)

#plt.tick_params(axis='both', which='both', bottom=False,top=False,labelbottom=False,right=False,left=False,labelleft=False)

f, axes = plt.subplots(12, 3, figsize=(6, 8), dpi=300)
axes = axes.flat


f.tight_layout()

print(name)

for i, column in enumerate(list(data)):
    axes[i].tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    axes[i].set_facecolor('#D8E1E6')
    sns.lineplot(data=data.loc[:, column], ax=axes[i])
    axes[i].set_title(column, loc='left', fontsize=9, pad = 3)
    axes[i].axvline(x=start, color='black')
plt.subplots_adjust(wspace=0.05, hspace=0.8)
f.savefig('./figure.png')
plt.close(f)  
 
