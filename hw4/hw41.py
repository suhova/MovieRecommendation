import pygad
import pandas as pd

df = pd.read_csv('27.txt', delimiter=' ', engine='python')
maxValues = df.columns.tolist()
maxW = int(maxValues[0])
maxV = int(maxValues[1])
df.columns = ['v', 'c']
df['w'] = df.index
data = df[['w','v','c']].values.tolist()

def fitness_func(solution, solution_idx):
    w = 0
    v = 0
    c = 0
    for (array, item) in zip(solution, data):
        if array > 0:
            w += item[0]
            v += item[1]
            c += item[2]
    if w > maxW or v > maxV:
        c = 0
    return c

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       sol_per_pop=10,
                       num_genes=30,
                       init_range_low=-1,
                       init_range_high=1
                       )

ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()

result = []
for i in range(0, 30):
    if solution[i] > 0:
        result.append(data[i])
result_df = pd.DataFrame.from_records(result, columns=['w','v','c'])

print(solution_fitness)
print(result_df)
result_df.to_csv('result41.csv', index=False)
