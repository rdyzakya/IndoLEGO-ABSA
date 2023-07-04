from itertools import combinations
import json

## HEURISTIK JENIS-JENIS TASK

# Existing task in papers: ATE, OTE, AOPE, E2E-ABSA/UABSA, ACSA, ASTE, TASD, ASQP

# all_tasks = ['a', 'c', 'o', 's', 'ac', 'ao', 'as', 'co', 'cs', 'os', 'aco', 'acs', 'aos', 'cos', 'acos']
all_tasks = ['ao', 'as', 'cs', 'aos', 'acs', 'acos']

extraction_task = all_tasks.copy()
# extraction_task.remove('c') # tidak termasuk objektif absa
# extraction_task.remove('s') # ini masuknya nanti dokumen level dong

# concern pada imputation: perbedaan jumlah tuple antara simple task dan complex task
imputation_task = {
    # 'acos' : ['aos','aco'],
    'acos' : ['aos'],
    # aco --> bisa, karena hanya melengkapi hubungan a dan o, sehingga tidak akan ada perbedaan jumlah tuple
    # acs --> tidak bisa, sebuah a bisa memiliki multiple opinion
    # aos --> bisa, hanya melengkapi kategori dari a
    # cos --> tidak bisa, karena bisa saja terdapat beberapa a dengan kategori yang sama dan memiliki nilai o dan s yang kebetulan sama
    # 'acs' : ['as'],
    # ac --> tidak bisa, karena ada kemungkinan perbedaan jumlah tuple, bisa saja sebuah a mmeiliki multiple s
    # as --> bisa, karena hanya melengkapi kategori a
    # cs --> tidak bisa, karena bisa saja ada beberapa a yang memiliki c yang sama dan s yang sama
    'aos' : ['ao'],
    'as' : ['a'],
    'ao' : ['a']
    # ao --> bisa, karena hanya melengkapi hubungan antara a dan o
    # as --> tidak bisa, karena bisa saja ada perbedaan jumlah tuple, ada beberapa o yang menjelaskan sebuah a
    # os --> tidak bisa, karena bisa saja ada beberapa a yang memiliki o dan s yang sama
    # 'ac' : ['a'],
    # a --> bisa, karena hanya melengkapi nilai c
    # c --> tidak bisa, karena bisa saja terdapat multiple a yang memiliki c sama. Pun, c sudah dieliminasi dari ekstraksi
    # 'os' : ['o']
    # o --> bisa, karena hanya melengkapi nilai s dari o (DEBATABLE, bisa saja tergantung aspeknya bagaimana)
    # s --> tidak bisa, karena bisa saja terdapat multiple o yang memiliki s yang sama. Pun, s sudah dieliminasi dari ekstraksi

    # cs sudah pasti tidak masuk
}

imputation_task2 = []
for k, v in imputation_task.items():
    for simple_task in v:
        imputation_task2.append((k,simple_task))

## HEURISTIK STARTING POINT
# Objektif kita adalah bisa melakukan aos dan cs
# Maka, minimal starting pointnya harus mengenalkan 4 komponen tersebut (untuk ekstraksi). Mulai dari jumlah task paling sedikit, maka mulai dari task paling kompleks
# Untuk imputation, minimal model dikenalkan dengan paradigma imputation

possible_extraction_task = []
for n in range(1,len(extraction_task)+1):
    for combo_extraction in combinations(extraction_task,n):
        all_components = ''.join(combo_extraction)
        all_components = set(all_components)
        if all_components != set('acos'):
            continue
        possible_extraction_task.append(list(combo_extraction))

possible_imputation_task = []
for n in range(1,len(imputation_task2)+1):
    for combo_imputation in combinations(imputation_task2,n):
        possible_imputation_task.append(list(combo_imputation))

for i in range(len(possible_imputation_task)):
    possible_imputation = {}
    for el in possible_imputation_task[i]:
        if el[0] not in possible_imputation.keys():
            possible_imputation[el[0]] = [el[1]]
        else:
            possible_imputation[el[0]].append(el[1])
    possible_imputation_task[i] = possible_imputation

def sorting_key(combo_el):
    length = len(combo_el)
    sum_square = 0
    for el in combo_el:
        sum_square += len(el)**2
    return sum_square, length

possible_extraction_task = sorted(possible_extraction_task,key=sorting_key)
possible_imputation_task = sorted(possible_imputation_task,key=sorting_key)
possible_imputation_task = possible_imputation_task

with open("combination_task.json",'w') as writer:
    result = {
        "extraction" : {
            "n" : len(possible_extraction_task),
            "combination" : possible_extraction_task
        },
        "imputation" : {
            "n" : len(possible_imputation_task),
            "combination" : possible_imputation_task
        }
    }

    json.dump(result,writer)

## START TRAINING
res15_path = "data/absa/en/zhang/interim/interim_2/rest15"
res16_path = "data/absa/en/zhang/interim/interim_2/rest16"
data_config_path = "/raid/m13519061/ta/facebook-absa/configs/data_config.json"
with open(data_config_path,'r') as reader:
    data_config = json.load(reader)

# RES 15
data_config["train"]["path"] = res15_path + "/train.txt"
data_config["val"]["path"] = res15_path + "/dev.txt"
data_config["test"]["path"] = res15_path + "/test.txt"