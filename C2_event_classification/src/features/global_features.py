import numpy as np
import awkward as ak
import vector
from typing import Dict, Tuple, Optional, Union

def extract_global_features(
    events: ak.Array,
    apply_scaling: bool = False,
    scaling_stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> Union[Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]], Dict[str, np.ndarray]]:
    """
    Extract global event-level features.
    
    Parameters:
        events (ak.Array): Awkward array containing event data
        apply_scaling (bool): Whether to apply standardization
        scaling_stats (Dict[str, Tuple[float, float]]): Optional standardization parameters
        
    Returns:
        Union[Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]], Dict[str, np.ndarray]]:
            Dict of global features and optionally scaling stats
    """
    # Ensure vector behavior is enabled
    vector.register_awkward()
    
    result = {}
    
    # Extract thrust features
    thrust_features = {
        "thrust_x": events.Thrust_x,
        "thrust_y": events.Thrust_y,
        "thrust_z": events.Thrust_z
    }
    
    # Create thrust magnitude and error
    thrust = vector.arr({
        'x': events.Thrust_x,
        'y': events.Thrust_y,
        'z': events.Thrust_z
    })

    # Particle multiplicity
    thrust_features["n_particles"] = events.nParticle
    
    # Vertex multiplicity 
    thrust_features["n_vertices"] = events.nVertex
    
    # Count secondary vertices
    thrust_features["n_secondary_vertices"] = ak.sum(events.Vertex_isPV == 0, axis=1)
    
    # Calculate charged multiplicity
    thrust_features["n_charged"] = ak.sum(events.Particle_q != 0, axis=1)
    
    # Calculate total event energy
    thrust_features["total_energy"] = ak.sum(events.Particle_e, axis=1)
    
    # Calculate total event transverse momentum
    thrust_features["total_pt"] = ak.sum(events.Particle_pt, axis=1)
    
    # Primary vertex features
    pv_mask = events.Vertex_isPV == 1
    
    # Ensure we have at least one primary vertex
    if np.min(ak.to_numpy(ak.count(pv_mask, axis=1))) > 0:
        pv_ntracks = events.Vertex_ntracks[pv_mask]
        thrust_features["pv_ntracks"] = ak.firsts(pv_ntracks)
        
        pv_chi2 = events.Vertex_chi2[pv_mask]
        thrust_features["pv_chi2"] = ak.firsts(pv_chi2)
    
    # Standardise features if required
    if apply_scaling:
        if scaling_stats is None:
            scaling_stats = {}
            
        for field, data in thrust_features.items():
            if field in scaling_stats:
                mean, std = scaling_stats[field]
            else:
                # Convert to numpy array before calculating statistics
                np_data = ak.to_numpy(data)
                mean = np.mean(np_data)
                std = np.std(np_data)
                
            # Handle zero std
            if std == 0:
                std = 1.0
                
            thrust_features[field] = (data - mean) / std
            scaling_stats[field] = (mean, std)
    
    # Convert to numpy arrays
    for field, data in thrust_features.items():
        result[field] = ak.to_numpy(data)
    
    return (result, scaling_stats) if apply_scaling else result
