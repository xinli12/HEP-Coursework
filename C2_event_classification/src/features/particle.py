import numpy as np
import awkward as ak
import vector
from typing import Union, Tuple, Dict, Optional

def extract_and_pad_particle_features(
    events: ak.Array,
    max_particles: int = 25,
    apply_scaling: bool = False,
    scaling_stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> Union[Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]], Dict[str, np.ndarray]]:
    """
    Extracts, sorts, pads, and optionally standardises particle and vertex features.
    
    This function processes particle-level data, calculates derived features such as
    angle to thrust axis, and adds one-hot encoding for particle identification.
    It works in conjunction with extract_global_features from global_features.py and 
    extract_and_pad_vertex_features from vertex.py to provide a complete feature set
    for event classification.
    
    Parameters:
        events (ak.Array): Awkward array containing event data.
        max_particles (int): Maximum number of particles to keep per event.
        apply_scaling (bool): Whether to apply standardization to features.
        scaling_stats (Dict[str, Tuple[float, float]]): Optional feature-wise standardisation 
                                                        parameters (mean, std).
    
    Returns:
        Union[Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]], Dict[str, np.ndarray]]:
            - If apply_scaling is True: Tuple of (feature_dict, scaling_stats)
            - If apply_scaling is False: Just the feature_dict
            Where feature_dict contains padded arrays of shape (N_events, max_particles)
    """

    # Ensure vector behaviour is enabled
    vector.register_awkward()

    # Filter and zip particle features
    particles = ak.zip({
        "Particle_px": events.Particle_px,
        "Particle_py": events.Particle_py,
        "Particle_pz": events.Particle_pz,
        "Particle_pt": events.Particle_pt,
        "Particle_e": events.Particle_e,
        "Particle_q": events.Particle_q,
        "Particle_ID": events.Particle_ID,
        "Particle_orivtx_ind": events.Particle_orivtx_ind
    })
    particles = particles[particles.Particle_orivtx_ind != -999]

    # Compute angle to thrust axis
    thrust = vector.arr({
        'x': events.Thrust_x,
        'y': events.Thrust_y,
        'z': events.Thrust_z
    })
    p3 = ak.zip({
        'x': particles.Particle_px,
        'y': particles.Particle_py,
        'z': particles.Particle_pz
    }, with_name="Momentum3D")
    particles["Particle_angle_to_thrust"] = thrust.deltaangle(p3)

    # Add is_charged
    particles["is_charged"] = ak.values_astype(particles.Particle_q != 0, float)

    # Add one-hot encodings
    pid_map = {
        "is_kaon": 321,
        "is_pion": 211,
        "is_muon": 13,
        "is_electron": 11,
        "is_photon": 22
    }
    particles = one_hot_encode_pid(particles, pid_map)

    # Sort by descending pt
    particles = particles[ak.argsort(particles.Particle_pt, ascending=False)]

    # Store origin_indices before standardization
    origin_indices = particles.Particle_orivtx_ind

    # Standardise particle features if required
    if apply_scaling:
        if scaling_stats is None:
            scaling_stats = {}
        for field in particles.fields:
            # Skip standardizing integer index fields
            if field == "Particle_orivtx_ind" or field == "Particle_ID":
                continue
                
            if field in scaling_stats:
                mean, std = scaling_stats[field]
            else:
                mean = std = None
            particles[field], mean, std = standardise(particles[field], mean, std)
            scaling_stats[field] = (mean, std)

    # Particle features to pad
    particle_fields = {
        "Particle_px": -999.0,
        "Particle_py": -999.0,
        "Particle_pz": -999.0,
        "Particle_e": -999.0,
        "Particle_q": -999.0,
        "Particle_angle_to_thrust": -999.0,
        "is_charged": -999.0,
        "is_kaon": -999.0,
        "is_pion": -999.0,
        "is_muon": -999.0,
        "is_electron": -999.0,
        "is_photon": -999.0
    }

    result = {field: pad_and_fill(particles[field], max_particles, pad_val)
              for field, pad_val in particle_fields.items()}

    # --- Vertex feature extraction ---
    vertex_fields = ["isPV", "chi2", "ntracks", "m", "x", "y", "z"]
    vertices = ak.zip({field: events[f"Vertex_{field}"] for field in vertex_fields})
    
    # Use the saved integer indices for vertex lookups
    vertex_features = {}
    for field in vertex_fields:
        data = vertices[field][origin_indices]
        feature_name = f"Vertex_{field}"
        if apply_scaling:
            if feature_name in scaling_stats:
                mean, std = scaling_stats[feature_name]
            else:
                mean = std = None
            data, mean, std = standardise(data, mean, std)
            scaling_stats[feature_name] = (mean, std)
        vertex_features[feature_name] = pad_and_fill(data, max_particles, -999.0)
        result[feature_name] = vertex_features[feature_name]

    # Displacement from PV
    pv_mask = vertices.isPV == 1
    pv = vertices[pv_mask]
    pv_pos = ak.zip({
        'x': pv.x[:, 0],
        'y': pv.y[:, 0],
        'z': pv.z[:, 0]
    }, with_name="Vector3D")
    particle_vtx_pos = ak.zip({
        'x': vertices.x[origin_indices],
        'y': vertices.y[origin_indices],
        'z': vertices.z[origin_indices]
    }, with_name="Vector3D")
    pv_broadcast = ak.broadcast_arrays(particle_vtx_pos.x, pv_pos)[1]
    displacement = particle_vtx_pos - pv_broadcast
    displacement_mag = displacement.mag

    if apply_scaling:
        name = "displacement_from_PV"
        if name in scaling_stats:
            mean, std = scaling_stats[name]
        else:
            mean = std = None
        displacement_mag, mean, std = standardise(displacement_mag, mean, std)
        scaling_stats[name] = (mean, std)

    result["displacement_from_PV"] = pad_and_fill(displacement_mag, max_particles, -999.0)

    return (result, scaling_stats) if apply_scaling else result


def standardise(array, mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[ak.Array, float, float]:
    """
    Standardise an array with optional given mean and std.
    
    Parameters:
        array (ak.Array): Array to standardize.
        mean (Optional[float]): Mean value to use for standardization. If None, calculated from array.
        std (Optional[float]): Standard deviation to use for standardization. If None, calculated from array.
        
    Returns:
        Tuple[ak.Array, float, float]: Standardized array, mean, and standard deviation
    """
    if mean is None or std is None:
        # Flatten the array to compute statistics on all values
        flat_array = ak.flatten(array)
        # Now convert to numpy
        np_array = ak.to_numpy(flat_array)
        mean = np.mean(np_array)
        std = np.std(np_array)
        # Handle zero std
        if std == 0:
            std = 1.0
    return (array - mean) / std, mean, std

def pad_and_fill(array: ak.Array, length: int, pad_val: float) -> np.ndarray:
    """
    Pad an Awkward array and fill missing values.
    
    Parameters:
        array (ak.Array): Array to pad.
        length (int): Desired length after padding.
        pad_val (float): Value to use for padding.
        
    Returns:
        np.ndarray: Padded numpy array of shape (N_events, length)
    """
    padded = ak.pad_none(array, length, clip=True)
    filled = ak.fill_none(padded, pad_val)
    return ak.to_numpy(filled)

def one_hot_encode_pid(particles: ak.Array, pid_map: Dict[str, int]) -> ak.Array:
    """
    Add one-hot encoded PID features to the particles array.
    
    Parameters:
        particles (ak.Array): Awkward array containing particle data.
        pid_map (Dict[str, int]): Mapping from feature names to particle ID values.
        
    Returns:
        ak.Array: Particles array with added one-hot encoded fields
    """
    for name, pid in pid_map.items():
        particles = ak.with_field(particles, ak.values_astype(abs(particles.Particle_ID) == pid, float), name)
    return particles