import numpy as np
import awkward as ak
from typing import Dict, Tuple, Optional, Union

def extract_and_pad_vertex_features(
    events: ak.Array, 
    max_vertices: int = 4, 
    apply_scaling: bool = False,
    scaling_stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> Union[Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]], Dict[str, np.ndarray]]:
    """
    Extract and pad secondary vertex features with optional standardisation.

    Parameters:
        events (ak.Array): Awkward array containing vertex data.
        max_vertices (int): Maximum number of vertices per event.
        apply_scaling (bool): Whether to apply standardization to features.
        scaling_stats (dict[str, tuple[float, float]]): Optional feature-wise standardisation parameters.

    Returns:
        Union[Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]], Dict[str, np.ndarray]]:
            Feature arrays (N_events, max_vertices) and optionally scaling stats
    """
    is_secondary = events.Vertex_isPV == 0
    vertices = ak.zip({
        "m": events.Vertex_m,
        "chi2": events.Vertex_chi2,
        "ntracks": events.Vertex_ntracks,
        "x": events.Vertex_x,
        "y": events.Vertex_y,
        "z": events.Vertex_z
    })
    sv = vertices[is_secondary]

    fields = {
        "m": -999.0,
        "chi2": -999.0,
        "ntracks": -999.0,
        "x": -999.0,
        "y": -999.0,
        "z": -999.0
    }

    if apply_scaling and scaling_stats is None:
        scaling_stats = {}

    result = {}
    for field, pad_val in fields.items():
        data = sv[field]
        feature_name = f"Vertex_{field}"
        
        if apply_scaling:
            if feature_name in scaling_stats:
                mean, std = scaling_stats[feature_name]
            else:
                # Flatten the array first to handle jagged arrays
                flat_data = ak.flatten(data)
                # Then convert to numpy array for calculating statistics
                np_data = ak.to_numpy(flat_data)
                mean = np.mean(np_data)
                std = np.std(np_data)
                # Handle zero std
                if std == 0:
                    std = 1.0
                scaling_stats[feature_name] = (mean, std)
            
            data = (data - mean) / std
            
        padded = ak.pad_none(data, max_vertices, clip=True)
        filled = ak.fill_none(padded, pad_val)
        array = ak.to_numpy(filled)
        result[f"sv_{field}"] = array
        
    return (result, scaling_stats) if apply_scaling else result
