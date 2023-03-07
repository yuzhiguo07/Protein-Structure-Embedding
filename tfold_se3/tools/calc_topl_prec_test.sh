#!/bin/bash

data_dir="/apdcephfs/private_jonathanwu/Datasets/CATH-Decoys-20201231"
fas_dpath="${data_dir}/fasta.files"
pdb_dpath="${data_dir}/pdb.files.native"
npz_dpath="${data_dir}/npz.files.msa"
did_fpath="${data_dir}/dom_names_tst.txt"

while IFS= read -r dom_id; do
    fas_fpath="${fas_dpath}/${dom_id}.fasta"
    pdb_fpath="${pdb_dpath}/${dom_id}.pdb"
    npz_fpath="${npz_dpath}/${dom_id}.npz"
    prec=`python calc_topl_prec.py \
        --fas_fpath ${fas_fpath} --pdb_fpath ${pdb_fpath} --npz_fpath ${npz_fpath} | awk '{print $3}'`
    echo "${dom_id} ${prec}"
done < ${did_fpath}
