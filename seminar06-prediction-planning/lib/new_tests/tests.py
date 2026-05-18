import os
import torch


DIR = os.path.dirname(os.path.abspath(__file__))


def compare_data(x, y, key=''):
    if isinstance(x, dict):
        assert isinstance(y, dict), f'{key}: first type {type(x)}, second type {type(y)}'

        assert set(x.keys()) == set(y.keys()), f'{key}: first keys {x.keys()}, second keys {y.keys()}'
    
        for k in x.keys():
            compare_data(x[k], y[k], f'{key}/{k}')
    
    if isinstance(x, list):
        assert isinstance(y, list), f'{key}: first type {type(x)}, second type {type(y)}'
    
        assert len(x) == len(y), f'{key}: first len {len(x)}, second len {len(y)}'

        for i in range(len(x)):
            compare_data(x[i], y[i], f'{key}/{i}')
    
    if isinstance(x, tuple):
        assert isinstance(y, tuple), f'{key}: first type {type(x)}, second type {type(y)}'
    
        assert len(x) == len(y), f'{key}: first len {len(x)}, second len {len(y)}'

        for i in range(len(x)):
            compare_data(x[i], y[i], f'{key}/{i}')
    
    if isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor), f'{key}: first type {type(x)}, second type {type(y)}'

        assert torch.isclose(x, y, rtol=1e-3, atol=1e-3).all(), f'{key}: tensors not close'



def test_normalize_wrapper(normalize_wrapper_cls):
    with open(os.path.join(DIR, 'data/features'), 'rb') as f:
        features = torch.load(f, weights_only=False)
    
    with open(os.path.join(DIR, 'data/prediction'), 'rb') as f:
        prediction = torch.load(f, weights_only=False)

    anchors = normalize_wrapper_cls.extract_anchor_x_y_yaw(features)

    with open(os.path.join(DIR, 'data/anchors'), 'rb') as f:
        target_anchors = torch.load(f, weights_only=False)

    compare_data(anchors, target_anchors)

    transformed_features = normalize_wrapper_cls.transform_features_to_agent_space(features, *target_anchors)
    
    with open(os.path.join(DIR, 'data/transformed_features'), 'rb') as f:
        target_transformed_features = torch.load(f, weights_only=False)

    compare_data(transformed_features, target_transformed_features)
    
    transformed_predictions = normalize_wrapper_cls.transform_predictions_from_agent_space(prediction, *target_anchors)

    with open(os.path.join(DIR, 'data/transformed_predictions'), 'rb') as f:
        target_transformed_predictions = torch.load(f, weights_only=False)

    compare_data(transformed_predictions, target_transformed_predictions)
   
    print('Normalize wrapper test passed')


def test_agent_renderer(agent_renderer_cls):
    with open(os.path.join(DIR, 'data/transformed_features_for_renderer'), 'rb') as f:
        features = torch.load(f, weights_only=False)
    
    with open(os.path.join(DIR, 'data/agent_map'), 'rb') as f:
        target_agent_map = torch.load(f, weights_only=False)


    agent_map = agent_renderer_cls(size=400, resolution=0.25)(features)
    compare_data(agent_map, target_agent_map)
    print('Agent renderer test passed')


def test_roadgraph_renderer(roadgraph_renderer_cls):
    with open(os.path.join(DIR, 'data/transformed_features_for_renderer'), 'rb') as f:
        features = torch.load(f, weights_only=False)
    
    with open(os.path.join(DIR, 'data/roadgraph_map'), 'rb') as f:
        target_roadgraph_map = torch.load(f, weights_only=False)


    roadgraph_map = roadgraph_renderer_cls(size=400, resolution=0.25)(features)
    compare_data(roadgraph_map, target_roadgraph_map)
    print('Roadgraph renderer test passed')


def test_traffic_light_renderer(traffic_light_renderer_cls):
    with open(os.path.join(DIR, 'data/transformed_features_for_renderer'), 'rb') as f:
        features = torch.load(f, weights_only=False)
    
    with open(os.path.join(DIR, 'data/trl_map'), 'rb') as f:
        target_trl_map = torch.load(f, weights_only=False)


    trl_map = traffic_light_renderer_cls(size=400, resolution=0.25)(features)
    compare_data(trl_map, target_trl_map)
    print('Traffic light renderer test passed')


def test_pixel_transformer(pixel_transformer_cls):
    pixel_transformer = pixel_transformer_cls(400, 0.25)

    with open(os.path.join(DIR, 'data/pixel_transformer_xy'), 'rb') as f:
        x, y = torch.load(f, weights_only=False)

    
    ix, iy = pixel_transformer.to_clipped_pixel(x, y)

    result = {
        'to_pixel': pixel_transformer.to_pixel(x, y),
        'to_clipped_pixel': pixel_transformer.to_clipped_pixel(x, y),
        'is_valid_coords': pixel_transformer.is_valid_coords(x, y),
        'to_coord': pixel_transformer.to_coord(ix, iy)
    }

    with open(os.path.join(DIR, 'data/pixel_transformer_result'), 'rb') as f:
        target_result = torch.load(f)

    compare_data(result, target_result)
    print('Pixel transformer test passed')


def test_metrics(metrics_fn):
    with open(os.path.join(DIR, 'data/gt'), 'rb') as f:
        gt = torch.load(f, weights_only=False)
    gt['log_trajectory']['valid'][..., -2:] = False

    with open(os.path.join(DIR, 'data/prediction'), 'rb') as f:
        prediction = torch.load(f, weights_only=False)

    result = metrics_fn(prediction, gt)
    
    with open(os.path.join(DIR, 'data/metrics_result'), 'rb') as f:
        target_result = torch.load(f, weights_only=False)

    compare_data(result, target_result)
    print('Metrics test passed')