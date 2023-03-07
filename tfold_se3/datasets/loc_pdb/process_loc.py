import os
import shutil
import random

fasta_fpath = '/mnt/smile1/pdb/Deeploc/deeploc_data.fasta'
pdb_dpath = '/mnt/smile1/pdb/Deeploc/PDB'

# save path
valid_pdb_dpath = '/mnt/smile1/pdb/Deeploc/v_pdb'
bcx_fpath = '/mnt/smile1/pdb/Deeploc/sequence-clusters/bc-0.out'
chain_id_dpath = '/mnt/smile1/pdb/Deeploc'
label_dpath = '/mnt/smile1/pdb/Deeploc'
fasta_dpath = '/mnt/smile1/pdb/Deeploc/fasta'
valid_num = 1108
seed = 42


if not os.path.exists(valid_pdb_dpath):
    os.makedirs(valid_pdb_dpath)

if not os.path.exists(fasta_dpath):
    os.makedirs(fasta_dpath)


label_dict = {'Nucleus-U':0, 'Nucleus-M':0, 'Nucleus-S':0,
                'Cytoplasm-S':1, # Cytoplasm-Nucleus-U not included
                'Extracellular-S':2,
                'Mitochondrion-U':3, 'Mitochondrion-M':3, 'Mitochondrion-S':3,
                'Cell.membrane-M':4,
                'Endoplasmic.reticulum-M':5, 'Endoplasmic.reticulum-U':5, 'Endoplasmic.reticulum-S':5,
                'Plastid-U':6, 'Plastid-S':6, 'Plastid-M':6,
                'Golgi.apparatus-M':7, 'Golgi.apparatus-S':7, 'Golgi.apparatus-U':7,
                'Lysosome/Vacuole-M':8, 'Lysosome/Vacuole-S':8, 'Lysosome/Vacuole-U':8,
                'Peroxisome-M':9, 'Peroxisome-S':9, 'Peroxisome-U':9
            }

test_dict, non_test_dict, all_dict = {}, {}, {}
non_test_list = []


with open(fasta_fpath) as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if '>' in line:
            line_list = line[1:].split(' ')
            chain_id = line_list[0]
            label_str = line_list[1].strip()
            with open(os.path.join(fasta_dpath, '{}.fasta'.format(chain_id)), 'w') as w:
                w.write(line)
                w.write(lines[idx+1])
            if label_str not in label_dict:
                # print(line, 'can\'t find label in the label dict')
                continue
            label = label_dict[label_str]
            if 'test' in line:
                test_dict[chain_id] = label
            else:
                non_test_dict[chain_id] = label
                non_test_list.append(chain_id)
            all_dict[chain_id] = label
            if not os.path.exists(os.path.join(valid_pdb_dpath, '{}.pdb'.format(chain_id))):
                shutil.copyfile(os.path.join(pdb_dpath, '{}.pdb'.format(chain_id)), os.path.join(valid_pdb_dpath, '{}.pdb'.format(chain_id)))

with open(bcx_fpath, 'w') as w:
    for key in all_dict:
        w.write('{}\n'.format(key))

with open(os.path.join(chain_id_dpath, 'loc_chain_ids_tst.txt'), 'w') as w_txt:
    with open(os.path.join(label_dpath, 'loc_tst.csv'), 'w') as w_csv:
        for (key, value) in test_dict.items():
            w_txt.write('{}\n'.format(key))
            w_csv.write('{},{}\n'.format(key, value))

with open(os.path.join(chain_id_dpath, 'loc_chain_ids_trnval.txt'), 'w') as w_txt:
    with open(os.path.join(label_dpath, 'loc_trnval.csv'), 'w') as w_csv:
        for (key, value) in non_test_dict.items():
            w_txt.write('{}\n'.format(key))
            w_csv.write('{},{}\n'.format(key, value))

with open(os.path.join(chain_id_dpath, 'loc_chain_ids_all.txt'), 'w') as w_txt:
    with open(os.path.join(label_dpath, 'loc_all.csv'), 'w') as w_csv:
        for (key, value) in all_dict.items():
            w_txt.write('{}\n'.format(key))
            w_csv.write('{},{}\n'.format(key, value))

random.seed(seed)
random.shuffle(non_test_list)
non_test_len = len(non_test_list)
print(non_test_len) # 11085
print(len(test_dict)) # 2773
valid_list = non_test_list[:valid_num]
train_list = non_test_list[valid_num:]

with open(os.path.join(chain_id_dpath, 'loc_chain_ids_val.txt'), 'w') as w_txt:
    with open(os.path.join(label_dpath, 'loc_val.csv'), 'w') as w_csv:
        for key in valid_list:
            w_txt.write('{}\n'.format(key))
            w_csv.write('{},{}\n'.format(key, non_test_dict[key]))

with open(os.path.join(chain_id_dpath, 'loc_chain_ids_trn.txt'), 'w') as w_txt:
    with open(os.path.join(label_dpath, 'loc_trn.csv'), 'w') as w_csv:
        for key in train_list:
            w_txt.write('{}\n'.format(key))
            w_csv.write('{},{}\n'.format(key, non_test_dict[key]))








