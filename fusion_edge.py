'''
Created on December 21, 2021
@author: Tongya Zheng (tyzheng@zju.edu.cn)
'''
__author__ = "zhengtongya"

import math
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from tqdm import trange

from args_config import args_config
from data_util import load_data, load_graph, load_label_data
from fusion import SamplingFusion
from graph import NeighborFinder, make_label_data
from gumbel_alpha import GumbelGAN
from neighbor_loader import BiSamplingNFinder
from util import EarlyStopMonitor, RandEdgeSampler, set_logger, set_random_seed

#import numba

set_random_seed()

try:
    parser = args_config()
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# Arguments
if True:
    KSAMPLERS = 2
    FREEZE = args.freeze
    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    UNIFORM = args.uniform
    ALPHA = args.alpha
    USE_TIME = args.time
    AGG_METHOD = args.agg_method
    ATTN_MODE = args.attn_mode
    SEQ_LEN = NUM_NEIGHBORS // KSAMPLERS
    DATA = args.data
    TASK = args.task
    HARD = args.hard
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim

    # Model initialize
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device('cuda:{}'.format(GPU))

    import socket

    DEVICE_STR = f'{socket.gethostname()}-{device.index}'
    PARAM_STR = f'{NUM_LAYER}-{NUM_HEADS}-{NUM_NEIGHBORS}-{HARD}-{DROP_OUT}-{BATCH_SIZE}'
    GUMBEL_PATH = f'./sample_cache/{TASK}-{FREEZE}-{args.data}-gumbel-{HARD}.pth'
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{TASK}-{FREEZE}-{PARAM_STR}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'

    def get_checkpoint_path(epoch):
        return f'./ckpt/{args.prefix}-{TASK}-{DEVICE_STR}-{PARAM_STR}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'


# set up logger
if True:
    logger = set_logger()
    logger.info(args)


def eval_one_epoch(hint, dps, src, dst, ts, label):
    with torch.no_grad():
        dps = dps.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(len(src) / TEST_BATCH_SIZE)
        scores = []

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(s_idx + TEST_BATCH_SIZE, num_test_instance)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            prob_score = dps.forward(src_l_cut, dst_l_cut, ts_l_cut,
                                     NUM_NEIGHBORS).sigmoid()
            scores.extend(list(prob_score.cpu().numpy()))
        pred_label = np.array(scores) > 0.5
        pred_prob = np.array(scores)
    return accuracy_score(label, pred_label), average_precision_score(
        label,
        pred_label), f1_score(label,
                              pred_label), roc_auc_score(label, pred_prob)


# Load data and train val test split
if True:
    if TASK == "edge":
        edges, n_nodes, val_time, test_time = load_graph(DATA)
        g_df = edges[["from_node_id", "to_node_id", "timestamp"]].copy()
        g_df["idx"] = np.arange(1, len(g_df) + 1)
        g_df.columns = ["u", "i", "ts", "idx"]
    elif TASK == "node":
        edges, nodes = load_data(DATA, "format")
        n_nodes = len(nodes)
        # padding node is 0, so add 1 here.
        id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        g_df = edges[["from_node_id", "to_node_id", "timestamp"]].copy()
        g_df["idx"] = np.arange(1, len(edges) + 1)
        g_df.columns = ["u", "i", "ts", "idx"]
        val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

    if len(edges.columns) > 4:
        e_feat = edges.iloc[:, 4:].to_numpy()
        padding = np.zeros((1, e_feat.shape[1]))
        e_feat = np.concatenate((padding, e_feat))
    else:
        e_feat = np.zeros((len(g_df) + 1, NODE_DIM))

    if FREEZE:
        n_feat = np.zeros((n_nodes + 1, NODE_DIM))
    else:
        bound = np.sqrt(6 / (2 * NODE_DIM))
        n_feat = np.random.uniform(-bound, bound, (n_nodes + 1, NODE_DIM))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    ts_l = g_df.ts.values

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())

    # set_random_seed()

# set train, validation, test datasets
if True:
    valid_train_flag = (ts_l < val_time)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]

    train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
    val_rand_sampler = RandEdgeSampler(src_l, dst_l)
    test_rand_sampler = RandEdgeSampler(src_l, dst_l)

# set validation, test datasets
# set validation, test datasets
if True:
    if TASK == "edge":
        _, val_data, test_data = load_label_data(dataset=DATA)

        val_src_l = val_data.u.values
        val_dst_l = val_data.i.values
        val_ts_l = val_data.ts.values
        val_label_l = val_data.label.values

        test_src_l = test_data.u.values
        test_dst_l = test_data.i.values
        test_ts_l = test_data.ts.values
        test_label_l = test_data.label.values
    elif TASK == "node":
        # select validation and test dataset
        valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
        valid_test_flag = ts_l > test_time

        val_src_l, val_dst_l, val_ts_l, val_label_l = make_label_data(
            src_l, dst_l, ts_l, valid_val_flag, test_rand_sampler)
        test_src_l, test_dst_l, test_ts_l, test_label_l = make_label_data(
            src_l, dst_l, ts_l, valid_test_flag, test_rand_sampler)
    else:
        raise NotImplementedError(TASK)

# Initialize the data structure for graph and edge sampling
# build the graph for fast query
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l,
                              train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=True)

# # full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=True)

gumbel_gnn = GumbelGAN(full_ngh_finder,
                       n_feat,
                       e_feat,
                       n_feat_freeze=FREEZE,
                       num_layers=1,
                       use_time=USE_TIME,
                       agg_method=AGG_METHOD,
                       attn_mode=ATTN_MODE,
                       seq_len=SEQ_LEN,
                       n_head=1,
                       drop_out=DROP_OUT,
                       node_dim=NODE_DIM,
                       time_dim=TIME_DIM,
                       hard=HARD,
                       num_neighbors=NUM_NEIGHBORS)
gumbel_gnn.load_state_dict(torch.load(GUMBEL_PATH, map_location=device))
gumbel_gnn = gumbel_gnn.to(device)
gumbel_gnn.eval()
bi_finder = BiSamplingNFinder(full_adj_list,
                              DATA,
                              gumbel_gnn,
                              NUM_NEIGHBORS,
                              mode=TASK,
                              hard=HARD,
                              freeze=FREEZE)

dps = SamplingFusion(bi_finder,
                     n_feat,
                     e_feat,
                     k_samplers=2,
                     n_feat_freeze=FREEZE,
                     num_layers=NUM_LAYER,
                     use_time=USE_TIME,
                     agg_method=AGG_METHOD,
                     attn_mode=ATTN_MODE,
                     seq_len=SEQ_LEN,
                     n_head=NUM_HEADS,
                     drop_out=DROP_OUT,
                     node_dim=NODE_DIM,
                     time_dim=TIME_DIM)
optimizer = torch.optim.Adam(dps.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
dps = dps.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)

early_stopper = EarlyStopMonitor()
epoch_bar = trange(NUM_EPOCH)
for epoch in epoch_bar:
    # Training
    batch_bar = trange(num_batch)
    for k in batch_bar:
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut = train_src_l[s_idx:e_idx]
        dst_l_cut = train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        dps = dps.train()
        pos_prob, neg_prob = dps.contrast(src_l_cut, dst_l_cut, dst_l_fake,
                                          ts_l_cut, NUM_NEIGHBORS)

        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)

        loss.backward()
        optimizer.step()
        # get training results
        with torch.no_grad():
            dps = dps.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(),
                                         (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc = accuracy_score(true_label, pred_label)
            ap = average_precision_score(true_label, pred_label)
            f1 = f1_score(true_label, pred_label)
            auc = roc_auc_score(true_label, pred_score)
            batch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)

    # validation phase use all information
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', dps,
                                                      val_src_l, val_dst_l,
                                                      val_ts_l, val_label_l)
    epoch_bar.update()
    epoch_bar.set_postfix(acc=val_acc, f1=val_f1, auc=val_auc)

    if early_stopper.early_stop_check(val_auc):
        break
    else:
        torch.save(dps.state_dict(), get_checkpoint_path(epoch))

logger.info('No improvment over {} epochs, stop training'.format(
    early_stopper.max_round))
logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
best_model_path = get_checkpoint_path(early_stopper.best_epoch)
dps.load_state_dict(torch.load(best_model_path))
logger.info(
    f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
dps.eval()
# testing phase use all information
_, _, _, val_auc = eval_one_epoch('val for old nodes', dps, val_src_l,
                                  val_dst_l, val_ts_l, val_label_l)
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes',
                                                      dps, test_src_l,
                                                      test_dst_l, test_ts_l,
                                                      test_label_l)

logger.info('Test statistics: acc: {:.4f}, f1:{:.4f} auc: {:.4f}'.format(
    test_acc, test_f1, test_auc))

logger.info('Saving DPS model')
torch.save(dps.state_dict(), MODEL_SAVE_PATH)
logger.info('DPS models saved')

res_path = "results/{}-Fusion.csv".format(DATA)
headers = ["method", "dataset", "valid_auc", "accuracy", "f1", "auc", "params"]
if not os.path.exists(res_path):
    f = open(res_path, 'w+')
    f.write(",".join(headers) + "\r\n")
    f.close()
    os.chmod(res_path, 0o777)
config = f"hard={HARD},n_layer={NUM_LAYER},n_head={NUM_HEADS},time={USE_TIME},freeze={FREEZE},"
config += f"n_degree={NUM_NEIGHBORS},batch_size={BATCH_SIZE},dropout={DROP_OUT},"
config += f"lr={LEARNING_RATE}"
with open(res_path, "a") as file:
    file.write("Fusion,{},{:.4f},{:.4f},{:.4f},{:.4f},\"{}\"".format(
        DATA, val_auc, test_acc, test_f1, test_auc, config))
    file.write("\n")
