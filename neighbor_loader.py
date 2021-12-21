import logging

import numpy as np

from graph import NeighborFinder
from gumbel_alpha import GumbelNFinder


class BiSamplingNFinder(object):
    def __init__(self,
                 adj_list,
                 data,
                 gumbel_gnn,
                 num_neighbors,
                 mode="edge",
                 hard="soft",
                 freeze=False) -> None:
        # Binary sampling: first half consists of Time Decay Sampling, and the
        # second half consists of Gumbel Attention Sampling.
        super().__init__()
        self.logger = logging.getLogger(__name__)

        cache = np.load(f"sample_cache/{mode}-{data}-alpha.npz")
        ALPHA = cache["alpha"]
        # For each node, we compute an optimal alpha by the temporal link
        # repetition task.
        self.alpha_sampler = NeighborFinder(adj_list, exp=True, alpha=ALPHA)

        # For each node, gumbel_finder computes its most significant neighbors.
        self.gumbel_finder = GumbelNFinder(adj_list, gumbel_gnn, hard=hard)
        assert num_neighbors % 2 == 0, "Binary sampling requires the number is even."
        self.gumbel_finder.precompute(mode, data, num_neighbors // 2, freeze)

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors):
        assert len(src_idx_l) == len(cut_time_l)
        assert num_neighbors % 2 == 0, "Binary sampling requires the number is even."
        half_neighbors = num_neighbors // 2
        g_ngh, g_eidx, g_time = self.gumbel_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, half_neighbors)
        a_ngh, a_eidx, a_time = self.alpha_sampler.get_temporal_neighbor(
            src_idx_l, cut_time_l, half_neighbors)

        return (g_ngh, g_eidx, g_time), (a_ngh, a_eidx, a_time)
