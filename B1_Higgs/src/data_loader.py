import uproot
import awkward as ak
import matplotlib.pyplot as plt
import os

def download_data_files(path, file_list, destination_folder="../datasets"):
    """
    Download data files if they don't exist locally.
    
    Args:
        path: Base URL path for files
        file_list: List of filenames to download
        destination_folder: Local folder to save files
    """
    for fname in file_list:
        if not os.path.exists(f"{destination_folder}/{fname}"):
            os.makedirs(destination_folder, exist_ok=True)
            # For MacOS
            os.system(f"curl -o {destination_folder}/{fname} {path}/{fname}")
            # For Linux: uncomment the following line and comment the line above
            # os.system(f"wget -O {destination_folder}/{fname} {path}/{fname}")


def load_data_from_root_files(file_pattern, columns):
    """
    Load data from ROOT files into a Pandas DataFrame.
    
    Args:
        file_pattern: Pattern matching ROOT files to load
        columns: List of column names to load
        
    Returns:
        DataFrame containing the data
    """
    df = ak.to_dataframe(uproot.concatenate(file_pattern,
                                            filter_name=columns,
                                            library='ak'))
    return df

def plot_mass_histogram(df, mass_range=(100, 160), bins=100):
    """
    Plot histogram of diphoton invariant mass.
    
    Args:
        df: DataFrame with higgs_m column
        mass_range: Tuple with (min, max) range for plot
        bins: Number of bins for histogram
    """
    fig, ax = plt.subplots()

    # Create histogram with styling options
    counts, bins, patches = ax.hist(df['higgs_m'][:,0], 
                                   bins=bins,
                                   color='royalblue',
                                   alpha=0.7,
                                   edgecolor='black',
                                   linewidth=0.5)

    # Highlight the Higgs mass region around 125 GeV
    ax.axvspan(123, 127, alpha=0.2, color='red')
    ax.axvline(125, color='red', linestyle='--', alpha=0.8, label='$m_H$ = 125 GeV')

    ax.set_xlabel('$m_{\gamma\gamma}$ [GeV]')
    bin_width = bins[1] - bins[0]
    ax.set_ylabel(f'Events / {bin_width:.1f} GeV')

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()