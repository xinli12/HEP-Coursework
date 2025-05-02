import vector

def calculate_diphoton_invariant_mass(df):
    """
    Calculate the invariant mass of diphoton system and add it to the DataFrame.
    
    Args:
        df: DataFrame containing photon data
        
    Returns:
        DataFrame with higgs_m column added
    """
    # Make photon 4-vectors
    photon1_p4 = vector.arr({"pt": df['photon_pt'][:,0],
                             "eta": df['photon_eta'][:,0],
                             "phi": df['photon_phi'][:,0],
                             "E": df['photon_e'][:,0]})

    photon2_p4 = vector.arr({"pt": df['photon_pt'][:,1],
                             "eta": df['photon_eta'][:,1],
                             "phi": df['photon_phi'][:,1],
                             "E": df['photon_e'][:,1]})

    # Calculate invariant mass from sum of 4-vectors
    higgs_p4 = photon1_p4 + photon2_p4

    # Add mass to DataFrame
    df.loc[(slice(None), 0), 'higgs_m'] = higgs_p4.m
    df.loc[(slice(None), 1), 'higgs_m'] = higgs_p4.m
    
    return df

def select_diphoton_events(df, mass_range=(100, 160)):
    """
    Select events with exactly two photons and within mass range.
    
    Args:
        df: DataFrame with photon data
        mass_range: Tuple with (min, max) mass range in GeV
        
    Returns:
        DataFrame with events selection applied
    """
    # Select events with exactly two photons
    df = df.query('photon_n == 2').copy()
    
    # Create Higgs vectors and add mass information
    df = calculate_diphoton_invariant_mass(df)
    
    # Filter by invariant mass
    eff, df = apply_selection(df, f'higgs_m >= {mass_range[0]} and higgs_m < {mass_range[1]}')
    print(f"Mass range selection efficiency: {eff:.4f}")
    
    return df

def apply_photon_quality_cuts(df):
    """
    Apply quality cuts to photon data.
    
    Args:
        df: DataFrame with photon data
        
    Returns:
        DataFrame with quality cuts applied
    """
    # Make an explicit copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. pT cuts: leading > 50 GeV, subleading > 35 GeV
    passPT = (df['photon_pt'][:, 0] > 50) & (df['photon_pt'][:, 1] > 35)
    df.loc[(slice(None), 0), 'photon_passPT'] = passPT.values
    df.loc[(slice(None), 1), 'photon_passPT'] = passPT.values
    eff, df = apply_selection(df, 'photon_passPT')
    print(f'Photon pT efficiency: {eff:.4f}')

    # 2. Tight ID and isolation cuts
    passID = (df['photon_isTightID'][:, 0]) & (df['photon_isTightID'][:, 1])
    passIso = (df['photon_isTightIso'][:, 0]) & (df['photon_isTightIso'][:, 1])
    
    # Create columns using .loc on the explicit copy
    df.loc[(slice(None), 0), 'photon_passID'] = passID.values
    df.loc[(slice(None), 1), 'photon_passID'] = passID.values
    df.loc[(slice(None), 0), 'passIso'] = passIso.values
    df.loc[(slice(None), 1), 'passIso'] = passIso.values
    
    eff, df = apply_selection(df, 'photon_passID')
    print(f'Photon ID efficiency: {eff:.4f}')
    eff, df = apply_selection(df, 'passIso')
    print(f'Photon isolation efficiency: {eff:.4f}')

    # 3. Energy ratio cuts: pT/m_H > 0.35 for both photons
    df['photon_ptom'] = df['photon_pt'] / df['higgs_m']
    passEnergyRatio = (df['photon_ptom'][:,0] > 0.35) & (df['photon_ptom'][:,1] > 0.35)
    df.loc[(slice(None), 0), 'passEnergyRatio'] = passEnergyRatio.values
    df.loc[(slice(None), 1), 'passEnergyRatio'] = passEnergyRatio.values
    eff, df = apply_selection(df, 'passEnergyRatio')
    print(f'Energy ratio efficiency: {eff:.4f}')
    
    return df

def apply_selection(df, query):
    """
    Apply a selection query to the DataFrame and return efficiency.
    
    Args:
        df: DataFrame to apply selection to
        query: Query string to filter DataFrame
        
    Returns:
        tuple: (efficiency, filtered DataFrame)
    """
    evs_before = df.index.get_level_values('entry').nunique()
    df = df.query(query)
    evs_after = df.index.get_level_values('entry').nunique()
    eff = evs_after / evs_before
    return eff, df
