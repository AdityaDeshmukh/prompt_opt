import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
from collections import defaultdict

# read json files
# parser = argparse.ArgumentParser(description="Select algo")
# parser.add_argument("dir", type=str)
# parser.add_argument("algos", type=list)
# args = parser.parse_args()
# algos, dir = args.algos, args.dir

dir = "u"
# algos = ['drgo_huber_std']
# ALGOS = [fr'R-REBEL (Huber-std)'] 
algos = ['grpo', 'new_l1_scaled','l1_std', 'drgo_huber', 'drgo_huber_std']
ALGOS = ['GRPO', fr'R-REBEL ($\ell_1$)', fr'R-REBEL ($\ell_1$-std)', 'R-REBEL (Huber)', fr'R-REBEL (Huber-std)']
plt.xlabel('Content score')
plt.ylabel('Sentiment score')
plt.ylim(0, 100)
plt.xlim(0, 100)

for i,algo in enumerate(algos):
    # aggregate by lambda
    acc = defaultdict(lambda: [[], []])  # {lambda: [[contents...],[styles...]]}
    with open(f'/{dir}/ad11/prompt_opt/{algo}/test/output.json', 'r') as f:
        d = json.load(f)
        for lam, mc, ms in zip(d["lmbdas"], d["mean_contents"], d["mean_styles"]):
            if isinstance(lam, list):
                k = round(float(lam[0]), 3)  # stable key for equal-interval lambdas
            else:
                k = round(float(lam), 3)
            
            acc[k][0].append(mc)
            acc[k][1].append(ms)

    # averaged per-lambda points
    lambdas = np.array(sorted(acc.keys()))
    content = np.array([np.mean(acc[l][0]) for l in lambdas])
    style   = np.array([np.mean(acc[l][1]) for l in lambdas])

    # plot mean_style vs mean_content
    plt.plot(content, style, '.--', label=ALGOS[i])
    # plt.title(ALGOS[i])
    if False:
        for i in range(0,len(content)):
            # if i==0:
            #     x,y = -10, 15
            # elif i==1:
            #     x,y = -25, 0
            # else:
            x,y = -25, -10
            plt.annotate(fr'$\tau=${int(100*lambdas[i])}', (content[i], style[i]), # Text and coordinates of the point
                    textcoords="offset points", # How to position the text
                    xytext=(x, y), # Distance from text to points (x,y)
                    ha='center') # Horizontal alignment of the text
    
plt.legend()
plt.savefig('huber.png')