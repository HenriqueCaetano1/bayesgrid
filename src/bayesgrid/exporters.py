import os
import pandapower as pp
import networkx as nx
import pandas as pd
import numpy as np
import warnings

# We assume opendss_helpers.py is in the same directory
from .opendss_helpers import (
    carson_modificada_Zabc, 
    format_matrix_for_dss,
    PHASE_CODE_MAP,
    PHASE_TO_DSS_NODE,
    PHASE_TO_MATRIX_IDX
)

def save_synthetic_network(
    base_net, 
    path_power_phase, 
    path_frequency, 
    path_duration, 
    path_impedance, 
    output_path, 
    format='pandapower'
):
    """
    Loads all synthetic data samples from individual CSV files and
    exports them to either Pandapower or OpenDSS format.

    Args:
        base_net (pandapower.Network): The original pandapower network
            object that was used as the template.
        path_power_phase (str): File path to 'bus_power_and_phase_SAMPLES.csv'.
        path_frequency (str): File path to 'bus_frequency_SAMPLES.csv'.
        path_duration (str): File path to 'bus_duration_SAMPLES.csv'.
        path_impedance (str): File path to 'line_impedance_SAMPLES.csv'.
        output_path (str): The destination *folder* where the exported
            files will be saved.
        format (str): The target format: 'pandapower' or 'opendss'.
    """
    
    print("Loading synthetic data from specified paths...")
    try:
        df_power = pd.read_csv(path_power_phase)
        df_freq = pd.read_csv(path_frequency)
        df_dur = pd.read_csv(path_duration)
        df_imp = pd.read_csv(path_impedance)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file. {e}")
        print("Please ensure all file paths are correct.")
        return

    # Get a list of all unique sample_id's
    n_samples = df_power['sample_id'].max() + 1
    print(f"Found {n_samples} unique synthetic samples.")
    
    # Create the output directory
    os.makedirs(output_path, exist_ok=True)

    if format.lower() == 'pandapower':
        print(f"Exporting {n_samples} Pandapower networks to: {output_path}")
        _save_to_pandapower(base_net, df_power, df_freq, df_dur, df_imp, output_path, n_samples)
        print(f"\nPandapower export complete. {n_samples} files created.")
        
    elif format.lower() == 'opendss':
        print(f"Exporting {n_samples} OpenDSS networks to: {output_path}")
        graph = pp.topology.create_nxgraph(base_net)
        _save_to_opendss(base_net, graph, df_power, df_imp, output_path, n_samples)
        print(f"\nOpenDSS export complete. {n_samples} files created.")
        
    else:
        raise ValueError(f"Unknown format: '{format}'. Must be 'pandapower' or 'opendss'.")

# --- Private Helper for Pandapower ---
def _save_to_pandapower(net, df_power, df_freq, df_dur, df_imp, output_folder, n_samples):
    """
    Loops through every sample, creates a full pandapower network
    for each, and saves it as a separate .json file.
    """
    
    # We must ensure bus and line names exist for joining
    if 'name' not in net.bus.columns or net.bus['name'].isnull().any():
        net.bus['name'] = net.bus.index
    if 'name' not in net.line.columns or net.line['name'].isnull().any():
        net.line['name'] = net.line.index
        
    # Group data by sample for faster processing
    power_grouped = df_power.groupby('sample_id')
    freq_grouped = df_freq.groupby('sample_id')
    dur_grouped = df_dur.groupby('sample_id')
    imp_grouped = df_imp.groupby('sample_id')
    
    for s in range(n_samples):
        print(f"  ... generating pandapower sample {s} of {n_samples-1}", end="\r")
        
        net_synthetic = net.deepcopy()
        
        # Get data for this specific sample
        sample_power = power_grouped.get_group(s).set_index('bus_id')
        sample_freq = freq_grouped.get_group(s).set_index('bus_id')
        sample_dur = dur_grouped.get_group(s).set_index('bus_id')
        sample_imp = imp_grouped.get_group(s).set_index('line_id')
        
        # Join bus data (we join on 'name', which must match the 'bus_id' in the CSV)
        bus_data = net_synthetic.bus.join(sample_power, on='name')
        bus_data = bus_data.join(sample_freq['CAIFI_FIC'], on='name')
        bus_data = bus_data.join(sample_dur['CAIDI_DIC'], on='name')
        
        # Join line data (we join on 'name', which must match the 'line_id' in the CSV)
        line_data = net_synthetic.line.join(sample_imp, on='name')
        
        net_synthetic.bus = bus_data
        net_synthetic.line = line_data
        
        file_path = os.path.join(output_folder, f"net_sample_{s}.json")
        pp.to_json(net_synthetic, file_path)
    
    print("\n") # Newline after the progress indicator

# --- Private Helper for OpenDSS ---
def _save_to_opendss(net, graph, df_power, df_imp, output_folder, n_samples):
    """
    Loops through every sample and saves each one as a separate .dss file.
    """
    
    # Constants
    BASE_KV = net.bus.vn_kv.mean() 
    DISTANCIAS = [0.6, 0.6, 1.2] 
    SLACK_BUS_NAME = f"bus_{net.ext_grid.bus.iloc[0]}"
    P_Q_RATIO = 0.328
    
    # Group data for faster lookup
    power_grouped = df_power.groupby('sample_id')
    imp_grouped = df_imp.groupby('sample_id')

    # Ensure bus names are strings for matching
    if 'name' not in net.bus.columns or net.bus['name'].isnull().any():
        net.bus['name'] = net.bus.index
    net.bus['name'] = net.bus['name'].astype(str)
    
    for s in range(n_samples):
        print(f"  ... generating opendss sample {s} of {n_samples-1}", end="\r")
        
        sample_power = power_grouped.get_group(s).set_index('bus_id')
        sample_imp = imp_grouped.get_group(s).set_index('line_id')
        
        dss_commands = []
        dss_commands.append("clear")
        dss_commands.append(f"new circuit.synthetic_net basekv={BASE_KV} pu=1.0 phases=3")
        dss_commands.append(f"edit Vsource.source bus1={SLACK_BUS_NAME}.1.2.3")
        dss_commands.append("\n!---------------- LINES ----------------\n")
        
        for line_id, line in net.line.iterrows():
            from_bus_pp = line['from_bus']
            to_bus_pp = line['to_bus']
            
            from_bus_graph = str(net.bus.at[from_bus_pp, 'name'])
            to_bus_graph = str(net.bus.at[to_bus_pp, 'name'])

            # Determine Line Phases
            try:
                from_bus_phase_str = sample_power.at[from_bus_pp, 'phase']
                to_bus_phase_str = sample_power.at[to_bus_pp, 'phase']
            except KeyError:
                warnings.warn(f"Sample {s}, Line {line_id}: Bus name not found in power data. Skipping.")
                continue
                
            from_bus_phases, to_bus_phases = set(from_bus_phase_str), set(to_bus_phase_str)
            line_phases_set = from_bus_phases.intersection(to_bus_phases)

            if not line_phases_set:
                warnings.warn(f"Sample {s}, Line {line_id}: No common phases. Skipping.")
                continue

            line_phases_sorted = sorted(list(line_phases_set))
            num_phases = len(line_phases_sorted)

            # Get Impedance
            try:
                r1 = sample_imp.at[line_id, 'R1_ohm_per_km']
                x1 = sample_imp.at[line_id, 'X1_ohm_per_km']
            except KeyError:
                 warnings.warn(f"Sample {s}, Line {line_id}: Line name not found in impedance data. Skipping.")
                 continue
                 
            r1 = max(r1, 1e-6) # Enforce floor
            
            Zabc_full, _ = carson_modificada_Zabc(r1, x1, DISTANCIAS)

            # Subset Zabc matrix
            matrix_indices = [PHASE_TO_MATRIX_IDX[p] for p in line_phases_sorted]
            Zabc_subset = Zabc_full[np.ix_(matrix_indices, matrix_indices)]
            rmatrix_subset = np.real(Zabc_subset)
            xmatrix_subset = np.imag(Zabc_subset)

            # Create a unique LineCode
            linecode_name = f"LCode_{line_id}"
            r_str = format_matrix_for_dss(rmatrix_subset)
            x_str = format_matrix_for_dss(xmatrix_subset)
            
            dss_commands.append(
                f"new linecode.{linecode_name} nphases={num_phases} "
                f"rmatrix={r_str} xmatrix={x_str} units=km"
            )

            # Create the Line element
            line_name = f"line_{line_id}"
            dss_phase_nodes = "".join([PHASE_TO_DSS_NODE[p] for p in line_phases_sorted])
            from_bus_name = f"bus_{from_bus_graph}{dss_phase_nodes}"
            to_bus_name = f"bus_{to_bus_graph}{dss_phase_nodes}"
            line_len_km = line['length_km']

            dss_commands.append(
                f"new line.{line_name} bus1={from_bus_name} bus2={to_bus_name} "
                f"linecode={linecode_name} length={line_len_km} units=km"
            )
        
        dss_commands.append("\n!---------------- LOADS ----------------\n")
        
        # Create the Loads
        for bus_id, bus in net.bus.iterrows():
            bus_graph_id = str(bus['name'])
            if f"bus_{bus_graph_id}" == SLACK_BUS_NAME:
                continue
                
            try:
                bus_data = sample_power.loc[bus_id]
            except KeyError:
                warnings.warn(f"Sample {s}, Bus {bus_id}: Bus name not found in power data. Skipping load.")
                continue
                
            bus_phases = bus_data['phase']
            
            if 'A' in bus_phases and bus_data['P_A'] > 0:
                kw = bus_data['P_A']
                kvar = kw * P_Q_RATIO
                dss_commands.append(f"new load.load_{bus_graph_id}_A bus1=bus_{bus_graph_id}.1 Phases=1 kv={BASE_KV} kw={kw} kvar={kvar} vminpu=0.85")
            if 'B' in bus_phases and bus_data['P_B'] > 0:
                kw = bus_data['P_B']
                kvar = kw * P_Q_RATIO
                dss_commands.append(f"new load.load_{bus_graph_id}_B bus1=bus_{bus_graph_id}.2 Phases=1 kv={BASE_KV} kw={kw} kvar={kvar} vminpu=0.85")
            if 'C' in bus_phases and bus_data['P_C'] > 0:
                kw = bus_data['P_C']
                kvar = kw * P_Q_RATIO
                dss_commands.append(f"new load.load_{bus_graph_id}_C bus1=bus_{bus_graph_id}.3 Phases=1 kv={BASE_KV} kw={kw} kvar={kvar} vminpu=0.85")

        # Add Solve commands
        dss_commands.append("\n!---------------- SOLUTION ----------------\n")
        dss_commands.append("set controlmode=off")
        dss_commands.append(f"Set Voltagebases=[{BASE_KV}]")
        dss_commands.append("calcv")
        dss_commands.append("Solve")
        
        # Write the .dss file
        file_path = os.path.join(output_folder, f"master_sample_{s}.dss")
        with open(file_path, 'w') as f:
            f.write("\n".join(dss_commands))
            
    print("\n") # Newline after the progress indicator