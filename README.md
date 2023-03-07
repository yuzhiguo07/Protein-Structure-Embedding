
# Self-Supervised Pre-training for Protein Embeddings Using Tertiary Structures

## requirment
```
# See cuda version:
cat /usr/local/cuda/version.txt or nvcc -V
conda install -c dglteam dgl-cudaXX.X  (replace the XX.X with cuda version)
pip install torch==1.7.1
pip install dgl
pip install lmdb
pip install lie-learn
pip install boto3
pip install h5py
pip install Bio
pip install pyyaml
pip install matplotlib
```

## PDB data pre-processing
```
export PYTHONPATH="/path/to/tFold-SE3-loc"
export NB_THREADS=30
cd tfold_se3/datasets/rcsb_pdb
```
modify config.yaml: 
`pdb_dpath_raw`: the dpath contains all the pdb files 
`bcx_dpath`: the dpath contains a `bc-0.out` file including all pdbid
`hdf_dpath`: saving dpath for hdf5 files
`log_dpath`: log dpath
`fas_dpath`: dpath for saving fasta files, no use in this version.
`pdb_dpath_natv`: no use in this version.
```
python build_dataset_sample.py
```

<!-- ## pre-training
```
cd tfold_se3/experiments/ebm_struct
modify config.yaml 
CUDA_VISIBLE_DEVICES=0 python main.py --config_fname config.yaml

``` -->
<!-- ## fine-tunning

 -->


## sample feature

```
cd ../..
cd experiments/ebm_struct/
modify config_sample_feat.yaml
```
for `config_sample_feat.yaml`:
`pid_fpath_rcsb_tst`: .txt fpath including all pdbid. (same with `bc-0.out` file)
`save_feat_dpath_rcsb`: save dpath for the generated features
`mdl_dpath`: model dpath
other hyper-paremeters are not using in this sampling version

Since the original model was trained during the internship, the model provided here (dpath: `tFold-SE3-loc/pdb/task01/models/sample`) is Not one of the models introduced in the paper. But it may achieve similar results.

```
CUDA_VISIBLE_DEVICES=0 python main.py --config_fname config_sample_feat.yaml
```

## load the generated features
```
import torch
feat = torch.load('fpath/to/the/feature/file.pt')
```
the feat is a python dict including 5 keys: (['pdbid', 'seq', 'inter_residue_feat', 'residue_level_feat', 'protein_level_feat'])
where pdbid is the pdbid and the seq is the protein sequence.
`inter_residue_feat` is the structrue embedding shape = L * L * C.
`residue_level_feat` is the embedding after 1D pooling shape = L * C.
`inter_residue_feat` is the embedding after 2D pooling shape = C.
where C = 64 in this model.

Notes:
After severial experiments, we found that the use of `inter_residue_feat` (L * L * C) has the best effect, and the performance is gradually weakened after 1D and 2D pooling. Therefore, we suggest users to use our structrue embedding in 2D downstream networks, such as 2D cnn or graph neural network. 

