import os

try:
    os.remove('experiment_commands.txt')
except FileNotFoundError:
    pass

iterations = 25
algorithms = ['GOT', 'L2', 'L2-inv']

text = ""

for algo in algorithms:
    for seed in range(iterations):
        text += f'python3 -u main.py {algo} {seed} --iterations 1000 --regularize\n'
with open('experiment_commands.txt', 'a') as file:
    file.write(text)
