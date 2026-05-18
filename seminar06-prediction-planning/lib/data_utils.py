import jax
import dataclasses

import numpy as np
import torch
from torch.utils.data import IterableDataset

from waymax import dataloader
from waymax.datatypes.roadgraph import filter_topk_roadgraph_points



class WaymaxDataset(IterableDataset):
    def __init__(self, config):
        self.config = config
        self.counter = 0

    def __iter__(self):
        self.counter += 1
        yield from dataloader.simulator_state_generator(
                config=dataclasses.replace(
                    self.config,
                    shuffle_seed=self.counter,
                )
            )



def get_tensors_slice(tensors, start=None, end=None):
    if isinstance(tensors, torch.Tensor):
        if start is None:
            start = 0
        if end is None:
            end = tensors.shape[-1]
        assert end <= tensors.shape[-1]
        return tensors[..., start:end]
    return  {
        k: get_tensors_slice(v, start, end) for k, v in tensors.items()
    }




def scenario_to_features_gt(
    scenario,
    features_first_timestamp=0,  # check scenario['timestamp'] for this
    features_timestamps=11,
    gt_timestamps=30,
    map_points=256,
    device='cpu',
    agent_to_predict_mask=None
):
    if agent_to_predict_mask is None:
        agent_to_predict_mask = scenario.object_metadata.is_sdc
    all_tensors = jax.tree_util.tree_map(lambda x: torch.tensor(np.asarray(x)).to(device), scenario)

    feat_start = features_first_timestamp
    feat_end = features_first_timestamp + features_timestamps

    cropped_map = filter_topk_roadgraph_points(
        scenario.roadgraph_points,
        scenario.log_trajectory.xy[agent_to_predict_mask, feat_end - 1],
        topk=map_points
    )
    cropped_map_tensors = jax.tree_util.tree_map(lambda x: torch.tensor(np.asarray(x)).to(device), cropped_map)

    gt_start = feat_end

    features = {
        'log_trajectory': get_tensors_slice(all_tensors['log_trajectory'], feat_start, feat_end),
        'sim_trajectory': get_tensors_slice(all_tensors['sim_trajectory'], feat_start, feat_end),
        'log_traffic_light': get_tensors_slice(all_tensors['log_traffic_light'], feat_start, feat_end),
        'object_metadata': dict(**all_tensors['object_metadata']),
        'roadgraph_points': get_tensors_slice(cropped_map_tensors),
        'agent_to_predict_mask': torch.tensor(np.asarray(agent_to_predict_mask)).to(device)
    }

    gt = {
        'log_trajectory': get_tensors_slice(all_tensors['log_trajectory'], gt_start, gt_start + gt_timestamps),
        'object_metadata': dict(**all_tensors['object_metadata']),
        'agent_to_predict_mask': torch.tensor(np.asarray(agent_to_predict_mask)).to(device)
    }

    return features, gt