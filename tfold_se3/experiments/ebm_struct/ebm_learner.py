"""Learner for energy-based models - 2D/3D inputs."""

import os
import json
import logging
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from tfold_se3.utils.prof_utils import *  # pylint: disable=wildcard-import
from tfold_se3.experiments.ebm_struct.ebm_dataset import EbmDatasetConfig
from tfold_se3.experiments.ebm_struct.ebm_dataset import EbmDataset
from tfold_se3.experiments.ebm_struct.ebm_dataset import update_2d_inputs
from tfold_se3.experiments.ebm_struct.ebm_dataset import build_3d_inputs
from tfold_se3.experiments.ebm_struct.ebm_dataset import build_3ds_inputs
from tfold_se3.experiments.ebm_struct.utils import calc_gdt_ts
from tfold_se3.experiments.ebm_struct.utils import calc_lddt_ca
from tfold_se3.experiments.ebm_struct.utils import calc_noise_stds
from tfold_se3.experiments.ebm_struct.utils import project_3d_cords
from tfold_se3.models.cond_resnet import CondResnet, Prednet
# from tfold_se3.models.se3_trans import SE3Trans
# from tfold_se3.models.se3_trans_sep import SE3TransSep
from tfold_se3.tools.metric_recorder import MetricRecorder
from tfold_se3.tools.struct_checker import StructChecker
from tfold_se3.utils import get_rand_str
from tfold_se3.utils import report_progress
from tfold_se3.utils import parse_fas_file
from tfold_se3.utils import export_pdb_file


class EbmLearner():
    """Learner for energy-based models - 2D/3D inputs."""

    def __init__(self, config):
        """Constructor function."""

        # initialization
        self.config = config
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # determine the number of dimensions for each feature type
        self.n_dims_onht = self.config['n_dims_onht'] if self.config['use_onht'] else 0
        self.n_dims_penc = self.config['n_dims_penc'] if self.config['use_penc'] else 0
        self.n_dims_dist = self.config['n_dims_dist'] if self.config['use_dist'] else 0
        self.n_dims_angl = self.config['n_dims_angl'] if self.config['use_angl'] else 0

        # validate hyper-parameter configurations
        if self.config['input_frmt'] == '2d':
            assert self.config['model_class'] in ['CondResnet']
        elif self.config['input_frmt'] == '3d':
            assert self.config['model_class'] in ['SE3Trans']
        elif self.config['input_frmt'] == '3ds':
            assert self.config['model_class'] in ['SE3TransSep']
        else:
            raise ValueError('unrecognized input format: ' + self.config['input_frmt'])

        # setup random noise's standard deviation levels
        self.noise_stds = calc_noise_stds(
            config['noise_std_max'], config['noise_std_min'], config['n_noise_levls'])
        self.noise_stds_tr = torch.tensor(self.noise_stds, device=self.device)

        # setup JiZhi-related configuations
        inst_id_key = 'TJ_INSTANCE_ID'
        self.w_jizhi = inst_id_key in os.environ # gai
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        # if self.w_jizhi and self.config['write_to_ceph']:
        if self.config['write_to_ceph']:
            # self.mdl_dpath = os.path.join(self.config['mdl_dpath'], os.getenv(inst_id_key))
            self.mdl_dpath = os.path.join(self.config['mdl_dpath'], '00a1')
        else:
            self.mdl_dpath = os.path.join(curr_dir, 'models')
        self.pdb_dpath = os.path.join(curr_dir, 'pdb.files')  # write PDB files to the local disk

        # export configurations to a JSON file
        os.makedirs(self.mdl_dpath, exist_ok=True)
        jsn_fpath = os.path.join(self.mdl_dpath, 'config.json')
        with open(jsn_fpath, 'w') as o_file:
            json.dump(self.config, o_file, indent=4)


    @profile
    def train(self):
        """Train a EBM model via noise contrastive estimation."""

        # build data loaders from training & validation subsets
        data_loader_trn = self.__build_data_loader(subset='trn')
        data_loader_val = self.__build_data_loader(subset='val')
        n_iters_per_epoch = len(data_loader_trn)
        logging.info('# of samples in the training subset: %d', len(data_loader_trn))
        logging.info('# of samples in the validation subset: %d', len(data_loader_val))

        # build a model as EBM's energy function
        model_base, model_trgt = self.__build_models()
        optimizer, scheduler = self.__build_optimizer(model_base)

        # restore models from previously generated checkpoints
        model_base, model_trgt, optimizer, scheduler, idx_epoch_last = \
            self.__restore_snapshot(model_base, model_trgt, optimizer, scheduler)

        # re-compute the optimal base & target models' validation loss
        loss_base_opt, loss_trgt_opt = None, None
        pth_fpath_base_opt = os.path.join(self.mdl_dpath, 'model_base_opt.pth')
        pth_fpath_trgt_opt = os.path.join(self.mdl_dpath, 'model_trgt_opt.pth')
        if idx_epoch_last != -1:
            with torch.no_grad():
                idx_iter = n_iters_per_epoch * (idx_epoch_last + 1)
                loss_base_opt = self.__eval_impl(
                    model_base, data_loader_val, idx_epoch_last, idx_iter)
                loss_trgt_opt = self.__eval_impl(
                    model_trgt, data_loader_val, idx_epoch_last, idx_iter)

        # update the model through multiple epochs
        for idx_epoch in range(idx_epoch_last + 1, self.config['n_epochs']):
            # display the greeting message
            lrn_rate = self.config['lr_init'] if scheduler is None else scheduler.get_last_lr()[0]
            logging.info('starting the %d-th training epoch (LR: %.2e)', idx_epoch + 1, lrn_rate)

            # train the model
            model_base, model_trgt = self.__train_impl(
                model_base, model_trgt, data_loader_trn, optimizer, idx_epoch)
            if scheduler is not None:
                scheduler.step()  # update the learning rate scheduler

            # evaluate the model
            with torch.no_grad():
                idx_iter = n_iters_per_epoch * (idx_epoch + 1)
                loss_base = self.__eval_impl(model_base, data_loader_val, idx_epoch, idx_iter)
                loss_trgt = self.__eval_impl(model_trgt, data_loader_val, idx_epoch, idx_iter)
                logging.info('validation loss: %.4e (base) / %.4e (target)', loss_base, loss_trgt)

            # save the snapshot for fast recovery
            self.__save_snapshot(model_base, model_trgt, optimizer, scheduler, idx_epoch)

            # update the optimal base & target models
            if loss_base_opt is None or loss_base_opt > loss_base:
                loss_base_opt = loss_base
                self.save_model(model_base, pth_fpath_base_opt)
            if loss_trgt_opt is None or loss_trgt_opt > loss_trgt:
                loss_trgt_opt = loss_trgt
                self.save_model(model_trgt, pth_fpath_trgt_opt)


    @profile
    def sample(self, pth_fpath=None):
        """Sample 3D structures with a pre-trained EMB model, and then evaluate (if possible)."""

        # build a data loader from the test subset
        data_loader = self.__build_data_loader(subset='tst')
        logging.info('# of samples in the test subset: %d', len(data_loader))

        # restore the pre-trained EBM model
        if pth_fpath is None:
            if os.path.exists(self.config['pth_fpath']):
                pth_fpath = self.config['pth_fpath']
            else:
                pth_fpath = os.path.join(self.mdl_dpath, self.config['pth_fname'])
        model, _ = self.__build_models()
        model = self.restore_model(model, pth_fpath)
        model.eval()

        # restore previous sampling results, and remove imcomplete ones
        prot_ids_skip = set()
        log_fpath = os.path.join(self.mdl_dpath, 'lddt_vals.csv')
        if os.path.exists(log_fpath):
            pdb_infos_per_id = defaultdict(list)
            with open(log_fpath, 'r') as i_file:
                for i_line in i_file:
                    pdb_fpath, lddt_val = i_line.strip().split(',')
                    prot_id = '_'.join(os.path.basename(pdb_fpath).split('_')[:-1])
                    pdb_infos_per_id[prot_id].append((pdb_fpath, lddt_val))
            for prot_id, pdb_infos in pdb_infos_per_id.items():
                if len(pdb_infos) == self.config['batch_size_tst']:
                    prot_ids_skip.add(prot_id)
            with open(log_fpath, 'w') as o_file:
                for prot_id in prot_ids_skip:
                    for pdb_fpath, lddt_val in pdb_infos_per_id[prot_id]:
                        o_file.write('%s,%s\n' % (pdb_fpath, lddt_val))

        # run Langevin dynamics based sampling to generate decoys
        pdb_dpath_natv = self.config['pdb_dpath_%s' % self.config['data_source']]
        for inputs, core_data in data_loader:
            # initialization
            prot_id = core_data['id']
            aa_seq = core_data['seq']

            # skip if the current protein ID has been sampled before
            if prot_id in prot_ids_skip:
                continue

            # run Langevin dynamics based sampling
            logging.info('running LD-sampling for %s (seq-len = %d)', prot_id, len(aa_seq))
            with torch.no_grad():
                cord_tns = self.__refine_samples(model, inputs, core_data)

            # export refined 3D structures to PDB files
            n_smpls = cord_tns.shape[0]
            pdb_fpaths_decy = [None for _ in range(n_smpls)]
            for idx in range(n_smpls):
                pdb_fpaths_decy[idx] = os.path.join(
                    self.pdb_dpath, '%s_%s.pdb' % (prot_id, get_rand_str()))
                export_pdb_file(aa_seq, cord_tns[idx], pdb_fpaths_decy[idx])

            # evaluate refined 3D structures
            pdb_fpath_natv = os.path.join(pdb_dpath_natv, '%s.pdb' % prot_id)
            if os.path.exists(pdb_fpath_natv):
                lddt_vals_raw = [calc_lddt_ca(x, pdb_fpath_natv) for x in pdb_fpaths_decy]
                lddt_vals = np.array(lddt_vals_raw, dtype=np.float32)
                logging.info('lDDT-Ca (%s): %d (ndc) / %.4f (avg) / %.4f (max)',
                             prot_id, lddt_vals.size, np.mean(lddt_vals), np.max(lddt_vals))

            # export evaluation results to the LOG file
            with open(log_fpath, 'a') as o_file:
                for idx, pdb_fpath_natv in enumerate(pdb_fpaths_decy):
                    o_file.write('%s,%.4f\n' % (pdb_fpath_natv, lddt_vals[idx]))

    @profile
    def sample_feat(self, pth_fpath=None):
        """sample_feat 3D structures with a pre-trained EMB model, and then do downstream task prediction (if possible)."""
        data_loader_tst = self.__build_data_loader(subset='tst')
        logging.info('# of samples in the test subset: %d', len(data_loader_tst))  

        # build a model as EBM's energy function
        model_base, model_trgt = self.__build_models()
        model_pred_base, model_pred_trgt = self.__build_pred_models()
        optimizer, scheduler = self.__build_optimizer(model_base)

        # restore the pre-trained EBM model
        if pth_fpath is None:
            if os.path.exists(self.config['pth_fpath']):
                pth_fpath = self.config['pth_fpath']
            else:
                pth_fpath = os.path.join(self.mdl_dpath, self.config['pth_fname'])

        model_base = self.restore_model(model_base, 'model_trgt_opt.pth')
        model_trgt = self.restore_model(model_trgt, 'model_trgt_opt.pth')
        # restore models from previously generated checkpoints
        model_base, model_trgt, model_pred_base, model_pred_trgt, optimizer, scheduler, idx_epoch_last = \
            self.__restore_snapshot_finetune(model_base, model_trgt, model_pred_base, model_pred_trgt, optimizer, scheduler)

        with torch.no_grad():
            self.__eval_sample_feat_impl(model_base, model_pred_base, data_loader_tst, 0, 0, self.config['save_feat_dpath_rcsb'], data_mark='test')


    @profile
    def finetune(self, pth_fpath=None):
        """finetune 3D structures with a pre-trained EMB model, and then do downstream task prediction (if possible)."""

        # build data loaders from the train/valid/test subset
        data_loader_trn = self.__build_data_loader(subset='trn')
        data_loader_val = self.__build_data_loader(subset='val')
        data_loader_tst = self.__build_data_loader(subset='tst')

        n_iters_per_epoch = len(data_loader_trn)
        logging.info('# of samples in the training subset: %d', len(data_loader_trn))
        logging.info('# of samples in the validation subset: %d', len(data_loader_val))
        logging.info('# of samples in the test subset: %d', len(data_loader_tst))  

        # build a model as EBM's energy function
        model_base, model_trgt = self.__build_models()
        model_pred_base, model_pred_trgt = self.__build_pred_models()
        optimizer, scheduler = self.__build_optimizer(model_base)
        # model, _ = self.__build_models()

        # restore the pre-trained EBM model
        if pth_fpath is None:
            if os.path.exists(self.config['pth_fpath']):
                pth_fpath = self.config['pth_fpath']
            else:
                pth_fpath = os.path.join(self.mdl_dpath, self.config['pth_fname'])
        if not os.path.exists(os.path.join(self.mdl_dpath, 'snapshot_finetune.pth')):
            # model_base = self.restore_model(model_base, 'model_base_opt.pth')
            model_base = self.restore_model(model_base, 'model_trgt_opt.pth')
            model_trgt = self.restore_model(model_trgt, 'model_trgt_opt.pth')
        # restore models from previously generated checkpoints
        model_base, model_trgt, model_pred_base, model_pred_trgt, optimizer, scheduler, idx_epoch_last = \
            self.__restore_snapshot_finetune(model_base, model_trgt, model_pred_base, model_pred_trgt, optimizer, scheduler)

        # re-compute the optimal base & target models' validation loss
        loss_base_opt, loss_trgt_opt = None, None
        acc_base_opt, acc_trgt_opt = None, None
        pth_fpath_base_opt = os.path.join(self.mdl_dpath, 'model_base_opt_finetune.pth')
        pth_fpath_trgt_opt = os.path.join(self.mdl_dpath, 'model_trgt_opt_finetune.pth')
        pth_fpath_base_pred_opt = os.path.join(self.mdl_dpath, 'model_base_pred_opt_finetune.pth')
        pth_fpath_trgt_pred_opt = os.path.join(self.mdl_dpath, 'model_trgt_pred_opt_finetune.pth')

        best_base_tst_acc = 0
        best_trgt_tst_acc = 0
        best_epoch = idx_epoch_last
        if idx_epoch_last != -1:
            with torch.no_grad():
                idx_iter = n_iters_per_epoch * (idx_epoch_last + 1)
                loss_base_opt, acc_base_opt = self.__eval_finetune_impl(
                    model_base, model_pred_base, data_loader_val, idx_epoch_last, idx_iter)
                loss_trgt_opt, acc_trgt_opt = self.__eval_finetune_impl(
                    model_trgt, model_pred_trgt, data_loader_val, idx_epoch_last, idx_iter)
                _, best_trgt_tst_acc = self.__eval_finetune_impl(
                    model_trgt, model_pred_trgt, data_loader_tst, idx_epoch_last, idx_iter, data_mark='test')
        
        # update the model through multiple epochs

        for idx_epoch in range(idx_epoch_last + 1, self.config['n_epochs']):
            # display the greeting message
            lrn_rate = self.config['lr_init'] if scheduler is None else scheduler.get_last_lr()[0]
            logging.info('starting the %d-th training epoch (LR: %.2e)', idx_epoch + 1, lrn_rate)

            # train the model
            model_base, model_trgt, model_pred_base, model_pred_trgt = self.__finetune_impl(
                model_base, model_trgt, model_pred_base, model_pred_trgt, data_loader_trn, optimizer, idx_epoch)
            
            if scheduler is not None:
                scheduler.step()  # update the learning rate scheduler

            # evaluate the model
            with torch.no_grad():
                idx_iter = n_iters_per_epoch * (idx_epoch + 1)
                loss_base, acc_base = self.__eval_finetune_impl(model_base, model_pred_base, data_loader_val, idx_epoch, idx_iter)
                loss_trgt, acc_trgt = self.__eval_finetune_impl(model_trgt, model_pred_trgt, data_loader_val, idx_epoch, idx_iter)
                logging.info('validation loss: %.4e (base) / %.4e (target)', loss_base, loss_trgt)
                logging.info('validation acc: %.4e (base) / %.4e (target)', acc_base, acc_trgt)
                loss_base_tst, acc_base_tst = self.__eval_finetune_impl(model_base, model_pred_base, data_loader_tst, idx_epoch, idx_iter, data_mark='test')
                loss_trgt_tst, acc_trgt_tst = self.__eval_finetune_impl(model_trgt, model_pred_trgt, data_loader_tst, idx_epoch, idx_iter, data_mark='test')
                logging.info('testing loss: %.4e (base) / %.4e (target)', loss_base_tst, loss_trgt_tst)
                logging.info('testing acc: %.4e (base) / %.4e (target)', acc_base_tst, acc_trgt_tst)
            # save the snapshot for fast recovery
            self.__save_snapshot_finetune(model_base, model_trgt, model_pred_base, model_pred_trgt, optimizer, scheduler, idx_epoch)

            # update the optimal base & target models
            # if loss_base_opt is None or loss_base_opt > loss_base:
            #     loss_base_opt = loss_base
            #     self.save_model(model_base, pth_fpath_base_opt)
            # if loss_trgt_opt is None or loss_trgt_opt > loss_trgt:
            #     loss_trgt_opt = loss_trgt
            #     self.save_model(model_trgt, pth_fpath_trgt_opt)
            if acc_base_opt is None or acc_base_opt > acc_base:
                acc_base_opt = acc_base
                self.save_model(model_base, pth_fpath_base_opt)
                best_base_tst_acc = acc_base_tst
                self.save_model(model_pred_base, pth_fpath_base_pred_opt)
                logging.info('======Best testing acc: %.4e on Epoch %d (base)', best_base_tst_acc, best_epoch)
                # best_epoch = idx_epoch

                self.save_model(model_pred_base, pth_fpath_base_pred_opt)
            if acc_trgt_opt is None or acc_trgt_opt > acc_trgt:
                acc_trgt_opt = acc_trgt
                self.save_model(model_trgt, pth_fpath_trgt_opt)
            # if acc_trgt_opt is None or acc_trgt_opt > acc_trgt:
                # acc_trgt_opt = acc_trgt
                best_trgt_tst_acc = acc_trgt_tst
                best_epoch = idx_epoch
                self.save_model(model_pred_trgt, pth_fpath_trgt_pred_opt)
                logging.info('======Best testing acc: %.4e on Epoch %d (target)', best_trgt_tst_acc, best_epoch)
            
            # input('debug')
        logging.info('Best testing acc: %.4e on Epoch %d (base)', best_base_tst_acc, best_epoch)
        logging.info('Best testing acc: %.4e on Epoch %d (target)', best_trgt_tst_acc, best_epoch)

    @classmethod
    def save_model(cls, model, path):
        """Save the model to a PyTorch checkpoint file."""

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        logging.info('model saved to %s', path)


    def restore_model(self, model, path):
        """Restore the model from a PyTorch checkpoint file."""

        # find the latest PyTorch checkpoint file, if not provided
        if not os.path.exists(path):
            logging.warning('checkpoint file (%s) does not exist; using the latest model ...', path)
            pth_fnames = [x for x in os.listdir(self.mdl_dpath) if x.endswith('.pth')]
            assert len(pth_fnames) > 0, 'no checkpoint file found under ' + self.mdl_dpath
            path = os.path.join(self.mdl_dpath, sorted(pth_fnames)[-1])

        # restore the model from a PyTorch checkpoint file
        model.load_state_dict(torch.load(path))
        logging.info('model restored from %s', path)

        return model


    def __save_snapshot(self, model_base, model_trgt, optimizer, scheduler, idx_epoch):
        """Save base & target models, optimizer, and LR scheduler to a checkpoint file."""

        snapshot = {
            'idx_epoch': idx_epoch,
            'model_base': model_base.state_dict(),
            'model_trgt': model_trgt.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': None if scheduler is None else scheduler.state_dict(),
        }
        pth_fpath = os.path.join(self.mdl_dpath, 'snapshot.pth')
        torch.save(snapshot, pth_fpath)
        logging.info('snapshot saved to %s', pth_fpath)

    def __save_snapshot_finetune(self, model_base, model_trgt, model_pred_base, model_pred_trgt, optimizer, scheduler, idx_epoch):
        """Save base & target models, optimizer, and LR scheduler to a checkpoint file."""

        snapshot = {
            'idx_epoch': idx_epoch,
            'model_base': model_base.state_dict(),
            'model_trgt': model_trgt.state_dict(),
            'model_pred_base': model_pred_base.state_dict(),
            'model_pred_trgt': model_pred_trgt.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': None if scheduler is None else scheduler.state_dict(),
        }
        pth_fpath = os.path.join(self.mdl_dpath, 'snapshot_finetune.pth')
        torch.save(snapshot, pth_fpath)
        logging.info('snapshot saved to %s', pth_fpath)

    def __restore_snapshot(self, model_base, model_trgt, optimizer, scheduler):
        """Restore base & target models, optimizer, and LR scheduler from the checkpoint file."""

        pth_fpath = os.path.join(self.mdl_dpath, 'snapshot.pth')
        if not os.path.exists(pth_fpath):
            idx_epoch = -1  # to indicate that no checkpoint file is available
        else:
            snapshot = torch.load(pth_fpath)
            logging.info('snapshot restored from %s', pth_fpath)
            idx_epoch = snapshot['idx_epoch']
            model_base.load_state_dict(snapshot['model_base'])
            model_trgt.load_state_dict(snapshot['model_trgt'])
            optimizer.load_state_dict(snapshot['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(snapshot['scheduler'])

        return model_base, model_trgt, optimizer, scheduler, idx_epoch

    def __restore_snapshot_finetune(self, model_base, model_trgt, model_pred_base, model_pred_trgt, optimizer, scheduler):
        """Restore base & target models, optimizer, and LR scheduler from the checkpoint file."""

        pth_fpath = os.path.join(self.mdl_dpath, 'snapshot_finetune.pth')
        if not os.path.exists(pth_fpath):
            idx_epoch = -1  # to indicate that no checkpoint file is available
        else:
            snapshot = torch.load(pth_fpath)
            logging.info('snapshot restored from %s', pth_fpath)
            idx_epoch = snapshot['idx_epoch']
            model_base.load_state_dict(snapshot['model_base'])
            model_trgt.load_state_dict(snapshot['model_trgt'])
            model_pred_base.load_state_dict(snapshot['model_pred_base'])
            model_pred_trgt.load_state_dict(snapshot['model_pred_trgt'])
            optimizer.load_state_dict(snapshot['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(snapshot['scheduler'])

        return model_base, model_trgt, model_pred_base, model_pred_trgt, optimizer, scheduler, idx_epoch


    def __build_data_loader(self, subset):
        """Build a data loader."""

        def _worker_init_fn(_):
            np.random.seed(torch.initial_seed() % 2 ** 32)

        # initialization
        assert subset in ['trn', 'val', 'tst'], 'unrecognized data subset: ' + subset

        # setup base configurations
        config_dict_base = {
            'source': self.config['data_source'],
            'input_frmt': self.config['input_frmt'],
            'hdf_dpath': self.config['hdf_dpath_%s' % self.config['data_source']],
            'npz_dpath': self.config['npz_dpath_%s' % self.config['data_source']],
            'pid_fpath': self.config['pid_fpath_%s_%s' % (self.config['data_source'], subset)],
            'batch_size': self.config['batch_size_%s' % subset],
            'n_dims_onht': self.config['n_dims_onht'] if self.config['use_onht'] else 0,
            'n_dims_penc': self.config['n_dims_penc'] if self.config['use_penc'] else 0,
            'n_dims_dist': self.config['n_dims_dist'] if self.config['use_dist'] else 0,
            'n_dims_angl': self.config['n_dims_angl'] if self.config['use_angl'] else 0,
            'filt_mthd': self.config['filt_mthd'],
            'pcut_vals': self.config['pcut_vals'],
            'pcnt_vals': self.config['pcnt_vals'],
            'seq_len_min': self.config['seq_len_min'],
            'seq_len_max': self.config['seq_len_max'],
        }
        if self.config['input_frmt'] == '2d':
            config_dict_base['n_dims_denc'] = self.config['n_dims_denc']
            config_dict_base['denc_mthd'] = self.config['denc_mthd']
            if subset in ['trn', 'val']:
                config_dict_base['crop_mode'] = self.config['crop_mode']
                config_dict_base['crop_size'] = self.config['crop_size']
        elif self.config['input_frmt'] == '3d':
            config_dict_base['dist_thres'] = self.config['dist_thres']
            config_dict_base['n_edges_max'] = self.config['n_edges_max']
        else:  # then self.config['input_frmt'] must be '3ds'
            config_dict_base['sep_list'] = self.config['sep_list']

        # setup subset-specific configurations
        if self.config['exec_mode'] == 'finetune' or self.config['exec_mode'] == 'sample_feat':
            config = EbmDatasetConfig(
                exec_mode='finetune',
                noise_stds=self.noise_stds,
                **config_dict_base,
            )
        elif subset in ['trn', 'val']:
            config = EbmDatasetConfig(
                exec_mode='train',
                noise_stds=self.noise_stds,
                **config_dict_base,
            )
        else:  # then <subset> must be 'tst'
            config = EbmDatasetConfig(
                exec_mode='sample',
                **config_dict_base,
            )

        # create a data loader from various sources
        dataset = EbmDataset(config)
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=True,
            num_workers=self.config['n_workers'], collate_fn=lambda x: x[0],
            pin_memory=True, worker_init_fn=_worker_init_fn,
        )

        return data_loader


    def __build_models(self):
        """Build a EBM model."""

        # build base & target models
        if self.config['model_class'] == 'CondResnet':
            model_base, model_trgt = self.__build_models_cr()
        elif self.config['model_class'] == 'SE3Trans':
            model_base, model_trgt = self.__build_models_st()
        else:  # then self.config['model_class'] must be 'SE3TransSep'
            model_base, model_trgt = self.__build_models_sts()

        # synchronize base & target models' parameters & BN statistics
        for param_base, param_trgt in zip(model_base.parameters(), model_trgt.parameters()):
            param_trgt.data.copy_(param_base.data)
        model_trgt.load_state_dict(model_base.state_dict())

        return model_base, model_trgt

    def __build_pred_models(self):
        """Build a EBM model."""

        # build pred model
        
        model_pred_base, model_pred_trgt = self.__build_models_pd()
        for param_base, param_trgt in zip(model_pred_base.parameters(), model_pred_trgt.parameters()):
            param_trgt.data.copy_(param_base.data)
        model_pred_trgt.load_state_dict(model_pred_base.state_dict())

        return model_pred_base, model_pred_trgt

    def __build_models_pd(self):
        """Build pred model with fc network."""

        # generate configurations for building the model
        config = {
            'n_chns_in': self.config['cr_n_chns_hid'],
            'n_chns_hid': self.config['pred_hid'],
            'n_cls': self.config['pred_cls'],
        }

        # build pred model
        pred_model_base = Prednet(**config).to(self.device)
        pred_model_trgt = Prednet(**config).to(self.device)
        # pred_model = Prednet(64, 32, 10).to(self.device)

        logging.info('prediction model initialized: %s', str(pred_model_base))

        return pred_model_base, pred_model_trgt


    def __build_models_cr(self):
        """Build EBM models with conditional residual network."""

        # determine the number of dimensions for input features
        n_chns_in = self.config['n_dims_denc'] + 2 * self.n_dims_onht \
            + 2 * self.n_dims_penc + self.n_dims_dist + self.n_dims_angl

        # generate configurations for building the model
        config = {
            'n_chns_in': n_chns_in,
            'n_chns_out': 1,
            'n_blocks': self.config['cr_n_blks'],
            'n_chns_hid': self.config['cr_n_chns_hid'],
            'block_type': self.config['cr_blk_type'],
            'norm_layer_type': self.config['cr_norm_lyr_type'],
            'norm_layer_depth': self.config['n_noise_levls'],
            'use_cc_attn': self.config['cr_use_cc_attn'],
        }

        # build base & target models
        model_base = CondResnet(**config).to(self.device)
        model_trgt = CondResnet(**config).to(self.device)
        logging.info('model initialized: %s', str(model_base))

        return model_base, model_trgt


    def __build_models_st(self):
        """Build EBM models with SE(3)-transformer."""

        # determine the number of dimensions for input node/edge features
        n_dims_node_in = self.n_dims_onht + self.n_dims_penc
        n_dims_edge = self.n_dims_dist + self.n_dims_angl

        # generate configurations for building the model
        config = {
            'n_dims_node_in': n_dims_node_in,
            'n_dims_node_out': self.config['st_n_dims_out'],
            'n_dims_edge': n_dims_edge,
            'n_blks': self.config['st_n_blks'],
            'n_degrees': self.config['st_n_dgrs'],
            'n_dims_node_hid': self.config['st_n_dims_hid'],
            'n_dims_graph_out': -1,
            'div_fctr': self.config['st_n_dfctr'],
            'n_heads': self.config['st_n_heads'],
            'cond_norm': self.config['st_cond_norm'],
            'cond_depth': self.config['n_noise_levls'],
            'pool_mthd': 'avg',
        }

        # build base & target models
        model_base = SE3Trans(**config).to(self.device)
        model_trgt = SE3Trans(**config).to(self.device)
        logging.info('model initialized: %s', str(model_base))

        return model_base, model_trgt


    def __build_models_sts(self):
        """Build EBM models with SE(3)-transformer-Sep."""

        # determine the number of dimensions for input node/edge features
        n_dims_node_in = self.n_dims_onht + self.n_dims_penc
        n_dims_edge = self.n_dims_dist + self.n_dims_angl

        # generate configurations for building the model
        config = {
            'n_dims_node_in': n_dims_node_in,
            'n_dims_node_out': self.config['st_n_dims_out'],
            'n_dims_edge': n_dims_edge,
            'sep_list': self.config['sep_list'],
            'n_blks': self.config['st_n_blks'],
            'n_degrees': self.config['st_n_dgrs'],
            'n_dims_node_hid': self.config['st_n_dims_hid'],
            'n_dims_graph_out': -1,
            'div_fctr': self.config['st_n_dfctr'],
            'n_heads': self.config['st_n_heads'],
            'cond_norm': self.config['st_cond_norm'],
            'cond_depth': self.config['n_noise_levls'],
            'pool_mthd': 'avg',
        }

        # build base & target models
        model_base = SE3TransSep(**config).to(self.device)
        model_trgt = SE3TransSep(**config).to(self.device)
        logging.info('model initialized: %s', str(model_base))

        return model_base, model_trgt


    def __build_optimizer(self, model):
        """Build a optimizer & its learning rate scheduler."""

        # create an Adam optimizer
        optimizer = Adam(
            model.parameters(), lr=self.config['lr_init'], weight_decay=self.config['weight_decay'])

        # create a LR scheduler
        if self.config['lr_scheduler'] == 'const':
            scheduler = None
        elif self.config['lr_scheduler'] == 'mstep':
            scheduler = MultiStepLR(
                optimizer, milestones=self.config['lr_mlstn'], gamma=self.config['lr_gamma'])
        elif self.config['lr_scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer, self.config['n_epochs'], eta_min=self.config['lr_min'])
        else:
            raise ValueError('unrecognized LR scheduler: ' + self.config['lr_scheduler'])

        return optimizer, scheduler


    @profile
    def __train_impl(self, model_base, model_trgt, data_loader, optimizer, idx_epoch):
        """Train the model - core implementation."""

        # train the model
        model_base.train()
        recorder = MetricRecorder()
        n_iters_per_epoch = len(data_loader)
        for idx_iter, (inputs, _) in enumerate(data_loader):
            # perform the forward pass w/ either 2D or 3D inputs
            if self.config['input_frmt'] == '2d':
                loss, metrics = self.__forward_2d(inputs, model_base)
            elif self.config['input_frmt'] == '3d':
                loss, metrics = self.__forward_3d(inputs, model_base)
            else:  # then self.config['input_frmt'] must be '3ds'
                loss, metrics = self.__forward_3ds(inputs, model_base)

            # perform the backward pass to update the base model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update the target model
            for param_base, param_trgt in zip(model_base.parameters(), model_trgt.parameters()):
                param_trgt.data.copy_(self.config['ema_alpha'] * param_trgt.data
                                      + (1.0 - self.config['ema_alpha']) * param_base.data)

            # record evaluation metrics
            recorder.add(metrics)
            if (idx_iter + 1) % self.config['n_iters_rep'] != 0:
                continue

            # report evaluation metrics periodically
            ratio = (idx_iter + 1) / n_iters_per_epoch
            recorder.display('Ep. #%d - %.2f%% (Train): ' % (idx_epoch + 1, 100.0 * ratio))
            if self.w_jizhi:
                idx_iter_full = n_iters_per_epoch * idx_epoch + idx_iter
                report_progress({'type': 'train', 'step': idx_iter_full, **recorder.get()})

        # synchronize base & target models' parameters & BN statistics
        model_trgt.load_state_dict(model_base.state_dict())

        # show final evaluation metrics at the end of epoch
        recorder.display('Ep. #%d - Final (Train): ' % (idx_epoch + 1))
        if self.w_jizhi:
            idx_iter_full = n_iters_per_epoch * (idx_epoch + 1)
            report_progress({'type': 'train', 'step': idx_iter_full, **recorder.get()})

        return model_base, model_trgt


    @profile
    def __eval_impl(self, model, data_loader, idx_epoch, idx_iter_full):
        """Evaluate the model - core implementation."""

        # evaluate the model
        model.eval()
        recorder = MetricRecorder()
        n_iters_per_epoch = len(data_loader)
        for idx_iter, (inputs, _) in enumerate(data_loader):
            # perform the forward pass w/ either 2D or 3D inputs
            if self.config['input_frmt'] == '2d':
                _, metrics = self.__forward_2d(inputs, model)
            elif self.config['input_frmt'] == '3d':
                _, metrics = self.__forward_3d(inputs, model)
            else:  # then self.config['input_frmt'] must be '3ds'
                _, metrics = self.__forward_3ds(inputs, model)

            # record evaluation metrics
            recorder.add(metrics)
            if (idx_iter + 1) % self.config['n_iters_rep'] != 0:
                continue

            # report evaluation metrics periodically
            ratio = (idx_iter + 1) / n_iters_per_epoch
            recorder.display('Ep. #%d - %.2f%% (Valid): ' % (idx_epoch + 1, 100.0 * ratio))

        # show final evaluation metrics at the end of epoch
        recorder.display('Ep. #%d - Final (Valid): ' % (idx_epoch + 1))
        if self.w_jizhi:
            report_progress({'type': 'test', 'step': idx_iter_full, **recorder.get()})

        return recorder.get()['Loss']

    @profile
    def __eval_sample_feat_impl(self, model, model_pred, data_loader, idx_epoch, idx_iter_full, save_dpath, data_mark='valid'):
        """Train the model - core implementation."""

        # train the model
        model.eval()
        model_pred.eval()
        if not os.path.exists(save_dpath):
            os.makedirs(save_dpath)
        n_iters_per_epoch = len(data_loader)
        for idx_iter, (inputs, data_dict) in enumerate(data_loader):
            # perform the forward pass w/ either 2D or 3D inputs
            if self.config['input_frmt'] == '2d':
                data_dict, encoder_feat = self.__forward_2d_sample_feat(inputs, model, model_pred, data_dict)
            elif self.config['input_frmt'] == '3d':
                _, metrics = self.__forward_3d(inputs, model)
            else:  # then self.config['input_frmt'] must be '3ds'
                _, metrics = self.__forward_3ds(inputs, model)
            
            pdbid = data_dict['id']
            seq = data_dict['seq']
            inter_residue_feat = encoder_feat.squeeze()
            # residue_level_feat = torch.nn.functional.avg_pool1d(inter_residue_feat, inter_residue_feat.size(-1)).squeeze().permute(1,0)
            residue_level_feat = torch.nn.functional.avg_pool1d(inter_residue_feat, inter_residue_feat.size(-1)).squeeze()
            protein_level_feat = torch.nn.functional.avg_pool2d(inter_residue_feat, (inter_residue_feat.size(-2), inter_residue_feat.size(-1))).squeeze()
            save_fpath = os.path.join(save_dpath, '{}.pt'.format(pdbid))
            save_dict = {'pdbid': pdbid,
                        'seq': seq, 
                        'inter_residue_feat': inter_residue_feat.detach().cpu(),
                        'residue_level_feat': residue_level_feat.detach().cpu(),
                        'protein_level_feat': protein_level_feat.detach().cpu()}
            torch.save(save_dict, save_fpath)
            print('{}\t/\t{}'.format(idx_iter, n_iters_per_epoch), end='\r')
        # return


    @profile
    def __eval_finetune_impl(self, model, model_pred, data_loader, idx_epoch, idx_iter_full, data_mark='valid'):
        """Train the model - core implementation."""

        # train the model
        model.eval()
        model_pred.eval()
        recorder = MetricRecorder()
        n_iters_per_epoch = len(data_loader)
        for idx_iter, (inputs, _) in enumerate(data_loader):
            # perform the forward pass w/ either 2D or 3D inputs
            if self.config['input_frmt'] == '2d':
                _, metrics = self.__forward_2d_finetune(inputs, model, model_pred)
            elif self.config['input_frmt'] == '3d':
                _, metrics = self.__forward_3d(inputs, model)
            else:  # then self.config['input_frmt'] must be '3ds'
                _, metrics = self.__forward_3ds(inputs, model)

            # record evaluation metrics
            recorder.add(metrics)
            if (idx_iter + 1) % self.config['n_iters_rep'] != 0:
                continue

            # report evaluation metrics periodically
            ratio = (idx_iter + 1) / n_iters_per_epoch
            recorder.display('Ep. #%d - %.2f%% (%s): ' % (idx_epoch + 1, 100.0 * ratio, data_mark))

        # show final evaluation metrics at the end of epoch
        recorder.display('Ep. #%d - Final (%s): ' % (idx_epoch + 1, data_mark))
        if self.w_jizhi:
            report_progress({'type': 'test', 'step': idx_iter_full, **recorder.get()})

        return recorder.get()['Loss'], recorder.get()['acc']

    @profile
    def __finetune_impl(self, model_base, model_trgt, model_pred_base, model_pred_trgt, data_loader, optimizer, idx_epoch):
        """Train the model - core implementation."""

        # train the model
        model_base.train()
        model_pred_base.train()
        
        recorder = MetricRecorder()
        n_iters_per_epoch = len(data_loader)
        for idx_iter, (inputs, _) in enumerate(data_loader):
            # perform the forward pass w/ either 2D or 3D inputs
            if self.config['input_frmt'] == '2d':
                loss, metrics = self.__forward_2d_finetune(inputs, model_base, model_pred_base)
            elif self.config['input_frmt'] == '3d':
                loss, metrics = self.__forward_3d(inputs, model_base)
            else:  # then self.config['input_frmt'] must be '3ds'
                loss, metrics = self.__forward_3ds(inputs, model_base)
            
            # perform the backward pass to update the base model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update the target model
            for param_base, param_trgt in zip(model_base.parameters(), model_trgt.parameters()):
                param_trgt.data.copy_(self.config['ema_alpha'] * param_trgt.data
                                      + (1.0 - self.config['ema_alpha']) * param_base.data)
            for param_pred_base, param_pred_trgt in zip(model_pred_base.parameters(), model_pred_trgt.parameters()):
                param_pred_trgt.data.copy_(self.config['ema_alpha'] * param_pred_trgt.data
                                      + (1.0 - self.config['ema_alpha']) * param_pred_base.data)

            # record evaluation metrics
            recorder.add(metrics)
            if (idx_iter + 1) % self.config['n_iters_rep'] != 0:
                continue

            # report evaluation metrics periodically
            ratio = (idx_iter + 1) / n_iters_per_epoch
            recorder.display('Ep. #%d - %.2f%% (Train): ' % (idx_epoch + 1, 100.0 * ratio))
            if self.w_jizhi:
                idx_iter_full = n_iters_per_epoch * idx_epoch + idx_iter
                report_progress({'type': 'train', 'step': idx_iter_full, **recorder.get()})

        # synchronize base & target models' parameters & BN statistics
        model_trgt.load_state_dict(model_base.state_dict())
        model_pred_trgt.load_state_dict(model_pred_base.state_dict())

        # show final evaluation metrics at the end of epoch
        recorder.display('Ep. #%d - Final (Train): ' % (idx_epoch + 1))
        if self.w_jizhi:
            idx_iter_full = n_iters_per_epoch * (idx_epoch + 1)
            report_progress({'type': 'train', 'step': idx_iter_full, **recorder.get()})

        return model_base, model_trgt, model_pred_base, model_pred_trgt

    @profile
    def __forward_2d(self, inputs, model):
        """Perform the forward pass with 2D inputs."""

        fmap_in = inputs['feat'].to(self.device)
        cond_idxs = inputs['idxs'].to(self.device)
        noise_stds = inputs['stds'].to(self.device)

        # scheme #1: apply scaling on estimated gradients for 2D distance
        #fmap_out = model(fmap_in, cond_idxs)
        #pred_tns = fmap_out / torch.reshape(noise_stds, [-1, 1, 1, 1])
        #loss, metrics = self.__calc_loss_2d(inputs, pred_tns)

        # scheme #2: apply scaling on estimated gradinets for 3D coordinates
        pred_tns, _ = model(fmap_in, cond_idxs)
        loss, metrics = self.__calc_loss_2d(inputs, pred_tns, noise_stds)

        return loss, metrics

    @profile
    def __forward_2d_sample_feat(self, inputs, model, model_pred, data_dict):
        """Perform the forward pass with 2D inputs."""
        # print(inputs.keys()) # dict_keys(['cond', 'feat', 'mask_p', 'mask_s', 'cord_p', 'cord_s', 'label', 'grad_p', 'grad_s', 'idxs', 'stds'])
        fmap_in = inputs['feat'].to(self.device)
        # print(inputs['feat'])
        cond_idxs = inputs['idxs'].to(self.device)
        noise_stds = inputs['stds'].to(self.device)

        
        # scheme #1: apply scaling on estimated gradients for 2D distance
        #fmap_out = model(fmap_in, cond_idxs)
        #pred_tns = fmap_out / torch.reshape(noise_stds, [-1, 1, 1, 1])
        #loss, metrics = self.__calc_loss_2d(inputs, pred_tns)

        # scheme #2: apply scaling on estimated gradinets for 3D coordinates
        _, encoder_feat  = model(fmap_in, cond_idxs)
        # pred_cls = model_pred(encoder_feat)
        # loss, metrics = self.__calc_loss_pred(inputs, pred_cls)


        return data_dict, encoder_feat

    @profile
    def __forward_2d_finetune(self, inputs, model, model_pred):
        """Perform the forward pass with 2D inputs."""

        fmap_in = inputs['feat'].to(self.device)
        cond_idxs = inputs['idxs'].to(self.device)
        noise_stds = inputs['stds'].to(self.device)

        # scheme #1: apply scaling on estimated gradients for 2D distance
        #fmap_out = model(fmap_in, cond_idxs)
        #pred_tns = fmap_out / torch.reshape(noise_stds, [-1, 1, 1, 1])
        #loss, metrics = self.__calc_loss_2d(inputs, pred_tns)

        # scheme #2: apply scaling on estimated gradinets for 3D coordinates
        _, encoder_feat  = model(fmap_in, cond_idxs)
        pred_cls = model_pred(encoder_feat)
        loss, metrics = self.__calc_loss_pred(inputs, pred_cls)


        return loss, metrics





    @profile
    def __forward_3d(self, inputs, model):
        """Perform the forward pass with 3D inputs."""

        graph = inputs['graph'].to(self.device)
        cond_idxs = inputs['idxs'].to(self.device)
        noise_stds = inputs['stds'].to(self.device)
        node_preds = model(graph, cond_idxs)
        pred_tns = torch.squeeze(node_preds['1'], dim=1) / torch.reshape(noise_stds, [-1, 1])
        loss, metrics = self.__calc_loss_3d(inputs, pred_tns)

        return loss, metrics


    @profile
    def __forward_3ds(self, inputs, model):
        """Perform the forward pass with 3DS inputs."""

        graph_dict = {k: v.to(self.device) for k, v in inputs['graph'].items()}
        cond_idxs = inputs['idxs'].to(self.device)
        noise_stds = inputs['stds'].to(self.device)
        node_preds = model(graph_dict, cond_idxs)
        pred_tns = torch.squeeze(node_preds['1'], dim=1) / torch.reshape(noise_stds, [-1, 1])
        loss, metrics = self.__calc_loss_3ds(inputs, pred_tns)

        return loss, metrics


    def __calc_loss_2d(self, inputs, pred_tns, noise_stds=None):
        """Calculate the loss & evaluation metrics for 2D inputs."""

        # initialization
        mask_mat_pri = inputs['mask_p'].to(self.device)
        mask_mat_sec = inputs['mask_s'].to(self.device)
        cord_tns_pri = inputs['cord_p'].to(self.device)
        cord_tns_sec = inputs['cord_s'].to(self.device)
        grad_tns_cord_pri = inputs['grad_p'].to(self.device)
        grad_tns_cord_sec = inputs['grad_s'].to(self.device)
        idxs_levl = inputs['idxs'].to(self.device)
        noise_stds = inputs['stds'].to(self.device)

        # recover gradients over 3D coordinates
        pgrd_tns_dist = torch.squeeze(pred_tns, dim=1) \
            * torch.unsqueeze(mask_mat_pri, dim=2) * torch.unsqueeze(mask_mat_sec, dim=1)
        pgrd_tns_cord_pri, pgrd_tns_cord_sec = \
            self.__recover_grad_cord(cord_tns_pri, cord_tns_sec, pgrd_tns_dist)

        # obtain 3D coordinates' ground-truth gradients and predicted ones
        mask_mat = torch.cat([mask_mat_pri, mask_mat_sec], axis=1)
        grad_tns_cord = torch.cat([grad_tns_cord_pri, grad_tns_cord_sec], axis=1)
        if noise_stds is None:
            pgrd_tns_cord = torch.cat([pgrd_tns_cord_pri, pgrd_tns_cord_sec], axis=1)
        else:
            pgrd_tns_cord_raw = torch.cat([pgrd_tns_cord_pri, pgrd_tns_cord_sec], axis=1)
            pgrd_tns_cord = pgrd_tns_cord_raw / torch.reshape(noise_stds, [-1, 1, 1])

        # calculate the loss & evaluation metrics
        loss, metrics = self.__calc_loss_impl(
            grad_tns_cord, pgrd_tns_cord, mask_mat, idxs_levl, noise_stds)

        return loss, metrics

    def __calc_loss_pred(self, inputs, pred_cls):
        """Calculate the loss & evaluation metrics for 2D inputs."""

        # initialization
        label = torch.tensor(int(inputs['label'])).to(self.device)
        label = torch.unsqueeze(label, 0)

        # calculate the loss & evaluation metrics
        loss, metrics = self.__calc_loss_pred_impl(
            pred_cls, label)

        return loss, metrics

    @classmethod
    def __recover_grad_cord(cls, cord_tns_pri, cord_tns_sec, grad_tns_dist):
        """Recover gradients over 3D coordinates from gradients over distance matrices."""
        grad_tns_cord_pri = 2 * (
            torch.unsqueeze(torch.sum(grad_tns_dist, dim=2), dim=2) * cord_tns_pri
            - torch.bmm(grad_tns_dist, cord_tns_sec)
        )
        grad_tns_cord_sec = 2 * (
            torch.unsqueeze(torch.sum(grad_tns_dist, dim=1), dim=2) * cord_tns_sec
            - torch.bmm(torch.transpose(grad_tns_dist, 1, 2), cord_tns_pri)
        )
        return grad_tns_cord_pri, grad_tns_cord_sec


    def __calc_loss_3d(self, inputs, pred_tns):
        """Calculate the loss & evaluation metrics for 3D inputs."""

        # initialization
        n_smpls = inputs['graph'].batch_size
        n_nodes = inputs['graph'].ndata['y'].shape[0] // n_smpls

        # obtain 3D coordinates' ground-truth gradients and predicted ones
        grad_tns = inputs['graph'].ndata['y'].view(n_smpls, n_nodes, 3).to(self.device)
        pred_tns_ext = pred_tns.view(n_smpls, n_nodes, 3)
        mask_mat = inputs['graph'].ndata['m'].view(n_smpls, n_nodes).to(self.device)
        idxs_levl = inputs['idxs'].view(n_smpls, n_nodes)[:, 0].to(self.device)
        noise_stds = inputs['stds'].view(n_smpls, n_nodes)[:, 0].to(self.device)

        # calculate the loss & evaluation metrics
        loss, metrics = self.__calc_loss_impl(
            grad_tns, pred_tns_ext, mask_mat, idxs_levl, noise_stds)

        return loss, metrics


    def __calc_loss_3ds(self, inputs, pred_tns):
        """Calculate the loss & evaluation metrics for 3DS inputs."""

        # initialization
        key_base = 'sep-1'
        n_smpls = inputs['graph'][key_base].batch_size
        n_nodes = inputs['graph'][key_base].ndata['y'].shape[0] // n_smpls

        # calculate the loss & evaluation metrics
        grad_tns = inputs['graph'][key_base].ndata['y'].view(n_smpls, n_nodes, 3).to(self.device)
        pred_tns_ext = pred_tns.view(n_smpls, n_nodes, 3)
        mask_mat = inputs['graph'][key_base].ndata['m'].view(n_smpls, n_nodes).to(self.device)
        idxs_levl = inputs['idxs'].view(n_smpls, n_nodes)[:, 0].to(self.device)
        noise_stds = inputs['stds'].view(n_smpls, n_nodes)[:, 0].to(self.device)
        loss, metrics = self.__calc_loss_impl(
            grad_tns, pred_tns_ext, mask_mat, idxs_levl, noise_stds)

        return loss, metrics


    @classmethod
    def __calc_loss_impl(cls, grad_tns, pred_tns, mask_mat, idxs_levl, noise_stds):
        """Calculate the loss & evaluation metrics - core implementation."""

        # loss function
        diff_tns = torch.reshape(noise_stds, [-1, 1, 1]) * (pred_tns - grad_tns)
        loss_vec = torch.mean(mask_mat * torch.square(torch.norm(diff_tns, dim=2)), dim=1)
        loss = torch.mean(loss_vec)

        '''
        # loss function - v2
        alpha = torch.reshape(noise_stds, [-1, 1, 1])
        pred_norm = torch.sqrt(torch.mean(torch.square(alpha * pred_tns)))
        grad_norm = torch.sqrt(torch.mean(torch.square(alpha * grad_tns)))
        loss_norm = torch.abs(pred_norm - grad_norm)
        loss_angl = torch.mean(torch.abs(
            torch.sum(pred_tns * grad_tns, dim=2) / torch.norm(pred_tns, dim=2) / torch.norm(grad_tns, dim=2) - 1.0))
        loss = loss_norm + loss_angl
        diff_tns = torch.reshape(noise_stds, [-1, 1, 1]) * (pred_tns - grad_tns)
        loss_vec = torch.mean(mask_mat * torch.square(torch.norm(diff_tns, dim=2)), dim=1)
        loss_lgc = torch.mean(loss_vec)
        '''

        # evaluation metrics
        mask_tns = torch.unsqueeze(mask_mat, dim=-1)
        mae = torch.mean(torch.abs(mask_tns * diff_tns))
        rmse = torch.sqrt(torch.mean(torch.square(mask_tns * diff_tns)))
        metrics = {
            'Loss': loss.item(),
            #'Loss-Norm': loss_norm.item(),
            #'Loss-Angle': loss_angl.item(),
            #'Loss-Legacy': loss_lgc.item(),
            #'Pred-Norm': pred_norm.item(),
            #'Grad-Norm': grad_norm.item(),
            'MAE': mae.item(),
            'RMSE': rmse.item(),
        }
        '''
        mask_mat = F.one_hot(idxs_levl, self.config['n_noise_levls']).float()
        mask_mat /= (torch.sum(mask_mat, dim=0, keepdim=True) + 1e-6)
        loss_per_idx = torch.matmul(
            mask_mat.T, loss_vec.unsqueeze(dim=-1)).squeeze(dim=1).detach().cpu().numpy()
        for idx in range(self.config['n_noise_levls']):
            metrics['Loss-%d' % idx] = loss_per_idx[idx]
        '''

        return loss, metrics


    @classmethod
    def __calc_loss_pred_impl(cls, pred_cls, label):
        """Calculate the loss & evaluation metrics - core implementation."""

        # loss function
        CEloss = torch.nn.CrossEntropyLoss()
        softmax = torch.nn.Softmax(dim=1)
        loss = CEloss(pred_cls, label)
        pred_cls_sm = softmax(pred_cls)
        pred_cls_idx = pred_cls_sm.argmax(dim=1)
        true_num = sum(pred_cls_idx == label)
        all_num = len(pred_cls_idx == label)
        acc = true_num/all_num

        '''
        # loss function - v2
        alpha = torch.reshape(noise_stds, [-1, 1, 1])
        pred_norm = torch.sqrt(torch.mean(torch.square(alpha * pred_tns)))
        grad_norm = torch.sqrt(torch.mean(torch.square(alpha * grad_tns)))
        loss_norm = torch.abs(pred_norm - grad_norm)
        loss_angl = torch.mean(torch.abs(
            torch.sum(pred_tns * grad_tns, dim=2) / torch.norm(pred_tns, dim=2) / torch.norm(grad_tns, dim=2) - 1.0))
        loss = loss_norm + loss_angl
        diff_tns = torch.reshape(noise_stds, [-1, 1, 1]) * (pred_tns - grad_tns)
        loss_vec = torch.mean(mask_mat * torch.square(torch.norm(diff_tns, dim=2)), dim=1)
        loss_lgc = torch.mean(loss_vec)
        '''

        # evaluation metrics
        # mask_tns = torch.unsqueeze(mask_mat, dim=-1)
        # mae = torch.mean(torch.abs(mask_tns * diff_tns))
        # rmse = torch.sqrt(torch.mean(torch.square(mask_tns * diff_tns)))
        # metrics = {
        #     'Loss': loss.item(),
        #     #'Loss-Norm': loss_norm.item(),
        #     #'Loss-Angle': loss_angl.item(),
        #     #'Loss-Legacy': loss_lgc.item(),
        #     #'Pred-Norm': pred_norm.item(),
        #     #'Grad-Norm': grad_norm.item(),
        #     'MAE': mae.item(),
        #     'RMSE': rmse.item(),
        # }

        metrics={'Loss': loss.item(),
            #   'true_num': true_num,
            #   'all_num': all_num,
              'acc': acc.item(),
        }

        '''
        mask_mat = F.one_hot(idxs_levl, self.config['n_noise_levls']).float()
        mask_mat /= (torch.sum(mask_mat, dim=0, keepdim=True) + 1e-6)
        loss_per_idx = torch.matmul(
            mask_mat.T, loss_vec.unsqueeze(dim=-1)).squeeze(dim=1).detach().cpu().numpy()
        for idx in range(self.config['n_noise_levls']):
            metrics['Loss-%d' % idx] = loss_per_idx[idx]
        '''

        return loss, metrics

    @profile
    def __refine_samples(self, model, inputs, core_data):
        """Refine samples via annealed Langevin dynamics."""

        # initialization
        aa_seq = core_data['seq']
        cord_mat_true_np = core_data['cord_t']
        cord_tns_pert_np = core_data['cord_p']
        batch_size, seq_len, _ = cord_tns_pert_np.shape

        # evalaute initial structures
        if self.config['eval_per_levl']:
            self.__evaluate_samples(aa_seq, cord_mat_true_np, cord_tns_pert_np)

        # visualize initial structures
        if self.config['enbl_arxiv_visual']:
            #self.__visualize_arxiv(0, cord_tns_pert_np)
            raise NotImplementedError

        # run Langevin dynamics based sampling process
        checker = StructChecker()
        scale = self.config['step_size_init']
        cord_tns_pert = None  # this will be initialized after the first iteration
        for idx_levl in range(self.config['n_noise_levls']):
            # calculate the step size
            noise_std = self.noise_stds[idx_levl]
            step_size = scale * (noise_std / self.noise_stds[-1]) ** 2
            noise_mult = self.config['noise_mult'] * np.sqrt(step_size)

            # generate conditional indices
            if self.config['input_frmt'] == '2d':
                cond_idxs = idx_levl \
                    * torch.ones((batch_size), dtype=torch.int64, device=self.device)
            else:
                cond_idxs = idx_levl \
                    * torch.ones((batch_size * seq_len), dtype=torch.int64, device=self.device)

            # run Langevin dynamics based sampling process at the current random noise level
            time_beg = timer()
            for idx_iter in range(self.config['n_iters_smp']):
                # update 2D/3D inputs
                if not (idx_levl == 0 and idx_iter == 0):
                    if self.config['input_frmt'] == '2d':
                        inputs = update_2d_inputs(inputs, core_data, cord_tns_pert)
                    elif self.config['input_frmt'] == '3d':
                        inputs = build_3d_inputs(core_data, cord_tns_pert.cpu().detach().numpy())
                    else:
                        inputs = build_3ds_inputs(core_data, cord_tns_pert.cpu().detach().numpy())

                # perform the forward pass w/ the score network
                if self.config['input_frmt'] == '2d':
                    fmap_in = inputs['feat'].to(self.device)
                    cord_tns_pert = inputs['cord_p'].to(self.device)
                    fmap_out = model(fmap_in, cond_idxs)

                    '''
                    # scheme #1: apply scaling on estimated gradients for 2D distance
                    grad_tns_dist = torch.squeeze(fmap_out, dim=1) / noise_std
                    grad_tns_cord_pri, grad_tns_cord_sec = \
                        self.__recover_grad_cord(cord_tns_pert, cord_tns_pert, grad_tns_dist)
                    grad_tns = (grad_tns_cord_pri + grad_tns_cord_sec) / 2.0
                    '''

                    # scheme #2: apply scaling on estimated gradinets for 3D coordinates
                    grad_tns_dist = torch.squeeze(fmap_out, dim=1)
                    grad_tns_cord_pri, grad_tns_cord_sec = \
                        self.__recover_grad_cord(cord_tns_pert, cord_tns_pert, grad_tns_dist)
                    grad_tns = (grad_tns_cord_pri + grad_tns_cord_sec) / 2.0 / noise_std
                elif self.config['input_frmt'] == '3d':
                    graph = inputs['graph'].to(self.device)
                    cord_tns_pert = torch.reshape(graph.ndata['x'], [batch_size, seq_len, -1])
                    node_preds = model(graph, cond_idxs)
                    grad_tns_raw = torch.squeeze(node_preds['1'], dim=1) / noise_std
                    grad_tns = torch.reshape(grad_tns_raw, [batch_size, seq_len, -1])
                else:  # then self.config['input_frmt'] must be '3ds'
                    graph_dict = {k: v.to(self.device) for k, v in inputs['graph'].items()}
                    cord_tns_pert = torch.reshape(
                        graph_dict['sep-1'].ndata['x'], [batch_size, seq_len, -1])
                    node_preds = model(graph_dict, cond_idxs)
                    grad_tns_raw = torch.squeeze(node_preds['1'], dim=1) / noise_std
                    grad_tns = torch.reshape(grad_tns_raw, [batch_size, seq_len, -1])

                # update 3D coordinates
                cord_tns_pert = cord_tns_pert \
                    + step_size / 2 * grad_tns \
                    + noise_mult * torch.randn(cord_tns_pert.shape, device=self.device)
                '''
                delt_tns_grad = step_size / 2 * grad_tns
                delt_tns_nois = noise_mult * torch.randn(cord_tns_pert.shape, device=self.device)
                cord_tns_pert = cord_tns_pert + delt_tns_grad + delt_tns_nois
                logging.info('coeff.: %.4e (grad) / %.4e (noise)', step_size / 2, noise_mult)
                logging.info('delta norm: %.4e (grad) / %.4e (noise)',
                             torch.norm(delt_tns_grad), torch.norm(delt_tns_nois))
                logging.info('SNR: %.4e', torch.norm(delt_tns_grad) / torch.norm(delt_tns_nois))
                '''

            # check whether there exists any handedness issues
            cord_tns_pert_np = cord_tns_pert.cpu().detach().numpy()
            for idx in range(batch_size):
                cord_mat_pert_np = cord_tns_pert_np[idx]
                is_valid = checker.check_cord_mat(cord_mat_pert_np, task='handedness')
                if not is_valid:
                    cord_tns_pert_np[idx] *= np.array([-1, 1, 1], dtype=np.float32)[None, :]
            cord_tns_pert = torch.tensor(cord_tns_pert_np, dtype=torch.float32, device=self.device)

            # evaluate refined structures
            if self.config['eval_per_levl']:
                self.__evaluate_samples(aa_seq, cord_mat_true_np, cord_tns_pert_np)

            # visualize refined structures at the end of current random noise level
            if self.config['enbl_arxiv_visual']:
                #self.__visualize_arxiv(idx_levl + 1, cord_tns_pert_np)
                raise NotImplementedError

            # estimate the remaining time to finish the sampling process
            if idx_levl == 0:
                eta = (self.config['n_noise_levls'] - 1) * (timer() - time_beg)
                logging.info('ETA: %.2f (s)', eta)

        return cord_tns_pert_np


    @classmethod
    def __evaluate_samples(cls, aa_seq, cord_mat_true, cord_tns_pred):
        """Evaluate sampled 3D coordinates."""

        # calculate the distance between adjacent CA-CA atoms
        dist_vals_list = []
        for idx in range(cord_tns_pred.shape[0]):
            cord_mat_pred = cord_tns_pred[idx]
            dist_vals = np.linalg.norm(cord_mat_pred[:-1] - cord_mat_pred[1:], axis=-1)
            dist_vals_list.append(dist_vals)
        dist_vals = np.concatenate(dist_vals_list)
        logging.info('CA-CA dist.: %.4f (avg) / %.4f (std)', np.mean(dist_vals), np.std(dist_vals))

        # calculate MAE & RMSE metrics for distance matrices
        mae_vals = []
        rmse_vals = []
        dist_mat_true = cdist(cord_mat_true, cord_mat_true, metric='euclidean')
        for idx in range(cord_tns_pred.shape[0]):
            cord_mat_pred = cord_tns_pred[idx]
            dist_mat_pred = cdist(cord_mat_pred, cord_mat_pred, metric='euclidean')
            mae_vals.append(np.mean(np.abs(dist_mat_true - dist_mat_pred)))
            rmse_vals.append(np.sqrt(np.mean(np.square(dist_mat_true - dist_mat_pred))))
        logging.info('MAE: %.4f / RMSE: %.4f', np.mean(mae_vals), np.mean(rmse_vals))

        # calculate GDT-TS & lDDT-Ca scores
        gdt_ts_vals = []
        lddt_ca_vals = []
        pdb_fpath_true = './outputs/native.pdb'
        pdb_fpath_pred = './outputs/decoy.pdb'
        export_pdb_file(aa_seq, cord_mat_true, pdb_fpath_true)
        for idx in range(cord_tns_pred.shape[0]):
            cord_mat_pred = cord_tns_pred[idx]
            export_pdb_file(aa_seq, cord_mat_pred, pdb_fpath_pred)
            gdt_ts_vals.append(calc_gdt_ts(pdb_fpath_pred, pdb_fpath_true))
            lddt_ca_vals.append(calc_lddt_ca(pdb_fpath_pred, pdb_fpath_true))
        logging.info('GDT-TS: %.4f / lDDT-Ca: %.4f', np.mean(gdt_ts_vals), np.mean(lddt_ca_vals))


    @classmethod
    def __visualize_arxiv(cls, idx_levl, cord_tns):
        """Visualize the distance matrix for generating figures in the arXiv submission."""

        # configurations
        dist_min = 0.0
        dist_max = 30.0
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        dat_dpath = os.path.join(curr_dir, 'arXiv_data')
        did_fpath = os.path.join(dat_dpath, 'dom_names.txt')
        with open(did_fpath, 'r') as i_file:
            prot_id = i_file.read().strip()
        fas_fpath = os.path.join(dat_dpath, '%s.fasta' % prot_id)
        pdb_fpath_natv = os.path.join(dat_dpath, '%s_native.pdb' % prot_id)
        pdb_fpath_decy = os.path.join(dat_dpath, '%s_decoy.pdb' % prot_id)
        png_dpath = os.path.join(dat_dpath, 'png.files.%s' % prot_id)

        # calculate each decoy's GDT-TS & lDDT-Ca scores, and then visualize its distance matrix
        gdt_ts_vals = []
        lddt_ca_vals = []
        _, aa_seq = parse_fas_file(fas_fpath)
        os.makedirs(png_dpath, exist_ok=True)
        for idx_decy in range(cord_tns.shape[0]):
            # obtain 3D coordinates & 2D distance matrix
            cord_mat = cord_tns[idx_decy]
            dist_mat = cdist(cord_mat, cord_mat, metric='euclidean')

            # calculate the lDDT-Ca score
            export_pdb_file(aa_seq, cord_mat, pdb_fpath_decy)
            gdt_ts_vals.append(calc_gdt_ts(pdb_fpath_decy, pdb_fpath_natv))
            lddt_ca_vals.append(calc_lddt_ca(pdb_fpath_decy, pdb_fpath_natv))

            # visualize the distance matrix
            image = np.clip(1.0 - (dist_mat - dist_min) / (dist_max - dist_min), 0.0, 1.0)
            np.fill_diagonal(image, 1.0)
            im = Image.fromarray((matplotlib.cm.Blues(image) * 255).astype(np.uint8))
            _, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(im)
            ax.axis('off')
            plt.tight_layout()
            png_fpath = os.path.join(png_dpath, '%d_%d.png' % (idx_levl + 1, idx_decy))
            plt.savefig(png_fpath)
            plt.close()

        # show all the decoys' GDT-TS & lDDT-Ca scores
        logging.info('GDT-TS @ %d => %s', idx_levl, ' '.join(['%.4f' % x for x in gdt_ts_vals]))
        logging.info('lDDT-Ca @ %d => %s', idx_levl, ' '.join(['%.4f' % x for x in lddt_ca_vals]))
