import argparse


def args_config():
    parser = argparse.ArgumentParser(
        'Interface for DPS experiments on link predictions')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='data sources to use',
                        default='ia-contact')
    parser.add_argument("-t",
                        "--task",
                        default="edge",
                        choices=["edge", "node"])
    parser.add_argument('-f', '--freeze', action='store_true')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix',
                        type=str,
                        default='Fusion',
                        help='prefix to name the checkpoints')
    parser.add_argument('--n_degree',
                        type=int,
                        default=20,
                        help='number of neighbors to sample')
    parser.add_argument('--n_head',
                        type=int,
                        default=2,
                        help='number of heads used in attention layer')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=50,
                        help='number of epochs')
    parser.add_argument('--n_layer',
                        type=int,
                        default=2,
                        help='number of network layers')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--drop_out',
                        type=float,
                        default=0.1,
                        help='dropout probability')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='idx for the gpu to use')
    parser.add_argument('--node_dim',
                        type=int,
                        default=128,
                        help='Dimentions of the node embedding')
    parser.add_argument('--time_dim',
                        type=int,
                        default=128,
                        help='Dimentions of the time embedding')
    parser.add_argument('--agg_method',
                        type=str,
                        choices=['attn', 'lstm', 'mean'],
                        help='local aggregation method',
                        default='attn')
    parser.add_argument('--attn_mode',
                        type=str,
                        choices=['prod', 'map'],
                        default='prod',
                        help='use dot product attention or mapping based')
    parser.add_argument('--time',
                        type=str,
                        choices=['time', 'pos', 'empty'],
                        help='how to use time information',
                        default='time')
    parser.add_argument('--uniform',
                        action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help="Sampling skewness.")
    parser.add_argument(
        "--hard",
        default="soft",
        choices=["soft", "hard", "atte"],
        help="hard Gumbel softmax",
    )
    parser.add_argument("--temp", default=1.0, type=float)
    parser.add_argument("--anneal", default=0.003, type=float)
    return parser


def node_args_config():
    parser = argparse.ArgumentParser(
        'Interface for DPS experiments on link predictions')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='data sources to use',
                        default='JODIE-wikipedia')
    parser.add_argument("-t", "--task", default="node", choices=["node"])
    parser.add_argument("--val_time", default=0.7, type=float)
    parser.add_argument("--node_layer", default=2, type=int)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Only use source_node embedding or use the combined embeddings.")
    parser.add_argument('--non-freeze',
                        dest='freeze',
                        action='store_false',
                        help='whether train the node embeddings')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix',
                        type=str,
                        default='Fusion',
                        help='prefix to name the checkpoints')
    parser.add_argument('--n_degree',
                        type=int,
                        default=20,
                        help='number of neighbors to sample')
    parser.add_argument('--n_head',
                        type=int,
                        default=2,
                        help='number of heads used in attention layer')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=50,
                        help='number of epochs')
    parser.add_argument('--n_layer',
                        type=int,
                        default=2,
                        help='number of network layers')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--drop_out',
                        type=float,
                        default=0.1,
                        help='dropout probability')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='idx for the gpu to use')
    parser.add_argument('--node_dim',
                        type=int,
                        default=128,
                        help='Dimentions of the node embedding')
    parser.add_argument('--time_dim',
                        type=int,
                        default=128,
                        help='Dimentions of the time embedding')
    parser.add_argument('--agg_method',
                        type=str,
                        choices=['attn', 'lstm', 'mean'],
                        help='local aggregation method',
                        default='attn')
    parser.add_argument('--attn_mode',
                        type=str,
                        choices=['prod', 'map'],
                        default='prod',
                        help='use dot product attention or mapping based')
    parser.add_argument('--time',
                        type=str,
                        choices=['time', 'pos', 'empty'],
                        help='how to use time information',
                        default='time')
    parser.add_argument('--uniform',
                        action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help="Sampling skewness.")
    parser.add_argument(
        "--hard",
        default="soft",
        choices=["soft", "hard", "atte"],
        help="hard Gumbel softmax",
    )
    parser.add_argument("--temp", default=1.0, type=float)
    parser.add_argument("--anneal", default=0.003, type=float)
    return parser


def gumbel_args_config():
    parser = argparse.ArgumentParser(
        'Interface for DPS experiments on link predictions')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='data sources to use',
                        default='ia-contact')
    parser.add_argument("-t",
                        "--task",
                        default="edge",
                        choices=["edge", "node"])
    parser.add_argument('-f', '--freeze', action='store_true')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix',
                        type=str,
                        default='',
                        help='prefix to name the checkpoints')
    parser.add_argument('--n_degree',
                        type=int,
                        default=20,
                        help='number of neighbors to sample')
    parser.add_argument('--n_head',
                        type=int,
                        default=1,
                        help='number of heads used in attention layer')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=50,
                        help='number of epochs')
    parser.add_argument('--n_layer',
                        type=int,
                        default=1,
                        help='number of network layers')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--drop_out',
                        type=float,
                        default=0.1,
                        help='dropout probability')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='idx for the gpu to use')
    parser.add_argument('--node_dim',
                        type=int,
                        default=128,
                        help='Dimentions of the node embedding')
    parser.add_argument('--time_dim',
                        type=int,
                        default=128,
                        help='Dimentions of the time embedding')
    parser.add_argument('--agg_method',
                        type=str,
                        choices=['attn', 'lstm', 'mean'],
                        help='local aggregation method',
                        default='attn')
    parser.add_argument('--attn_mode',
                        type=str,
                        choices=['prod', 'map'],
                        default='prod',
                        help='use dot product attention or mapping based')
    parser.add_argument('--time',
                        type=str,
                        choices=['time', 'pos', 'empty'],
                        help='how to use time information',
                        default='time')
    parser.add_argument('--uniform',
                        action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument(
        "--hard",
        default="soft",
        choices=["soft", "hard", "atte"],
        help="hard Gumbel softmax",
    )
    parser.add_argument("--temp", default=1.0, type=float)
    parser.add_argument("--anneal", default=0.003, type=float)
    return parser
