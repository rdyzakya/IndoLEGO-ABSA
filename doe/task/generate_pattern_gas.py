from itertools import permutations
import json

inputs = {
    "a" : "ASPECT",
    "c" : "CATEGORY",
    "o" : "OPINION",
    "s" : "SENTIMENT"
}

outputs = {
    "a" : "ASPECT",
    "c" : "CATEGORY",
    "o" : "OPINION",
    "s" : "SENTIMENT"
}

inputs_sep = ' , '
outputs_sep = ' , '

all_tasks = ['ao', 'as', 'cs', 'aos', 'acs', 'acos', 'a', 'c']

result = {task : [] for task in all_tasks}

for task in all_tasks:
    for permut in permutations(task):
        input_pattern = [inputs[se] for se in permut]
        output_pattern = [outputs[se] for se in permut]

        input_pattern = '( ' + inputs_sep.join(input_pattern) + ' )'
        output_pattern = '( ' + outputs_sep.join(output_pattern) + ' )'

        result[task].append({
            "input" : input_pattern,
            "output" : output_pattern
        })

with open("all_pattern_gas.json",'w') as fp:
    json.dump(result,fp)

print("DONE")