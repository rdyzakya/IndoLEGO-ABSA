from itertools import permutations
import json

inputs = {
    "a" : "aspect : <extra_id_0>",
    "c" : "category : <extra_id_1>",
    "o" : "opinion : <extra_id_2>",
    "s" : "sentiment : <extra_id_3>"
}

outputs = {
    "a" : "<extra_id_0> ASPECT",
    "c" : "<extra_id_1> CATEGORY",
    "o" : "<extra_id_2> OPINION",
    "s" : "<extra_id_3> SENTIMENT"
}

inputs_sep = ' , '
outputs_sep = ' '

all_tasks = ['ao', 'as', 'cs', 'aos', 'acs', 'acos', 'c', 'a']

result = {task : [] for task in all_tasks}

for task in all_tasks:
    for permut in permutations(task):
        input_pattern = [inputs[se] for se in permut]
        output_pattern = [outputs[se] for se in permut]

        input_pattern = inputs_sep.join(input_pattern)
        output_pattern = outputs_sep.join(output_pattern)

        result[task].append({
            "input" : input_pattern,
            "output" : output_pattern
        })

with open("all_pattern_lego.json",'w') as fp:
    json.dump(result,fp)

print("DONE")