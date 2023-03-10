import os

try:
    os.remove('experiment_commands.txt')
except FileNotFoundError:
    pass

iterations = 25
algorithms = ['GOT', 'L2', 'random', 'gw']

text = ""

for algo in algorithms:
    for seed in range(iterations):
        text += f'python3 -u graph_alignment.py {algo} {seed} --within_probability 0.7 --between_probability 0.1 --path ../results/07-01\n'
with open('experiment_commands.txt', 'a') as file:
    file.write(text)
