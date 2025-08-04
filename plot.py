import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
#read json file
content = []
style = []
parser = argparse.ArgumentParser(description="Select algo")
parser.add_argument("dir", type=str)
parser.add_argument("algo", type=str)

args = parser.parse_args()
# algo = 'kl'
# algo = 'grpo'
algo = args.algo
dir = args.dir
end = 0
directory_path = f'/{dir}/ad11/prompt_opt/{algo}/eval/'
for entry_name in os.listdir(directory_path):
    full_path = os.path.join(directory_path, entry_name)
    if os.path.isfile(full_path): # Check if it's a file
        k = int(''.join(filter(str.isdigit, entry_name)))
        if k > end:
            end = k

for i in range(end,end+1,50):
    with open(f'/{dir}/ad11/prompt_opt/{algo}/eval/outputs.step.{i}.json', 'r') as f:
        data = json.load(f)
        content.append(data['mean_contents'])
        style.append(data['mean_styles'])
content = np.array(content)
style = np.array(style)
plt.plot(content, style,'.')
plt.legend([f'$\lambda$={0.1*i:.1f}' for i in range(10)])
# plt.plot(content.T, style.T,'.')
plt.xlabel('Content score')
plt.ylabel('Sentiment score')
plt.ylim(0,100)
plt.xlim(0,100)
plt.title(f'{algo}')
plt.savefig(f'{algo}.png')