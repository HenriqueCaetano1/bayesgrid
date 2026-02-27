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

    elif format.lower() == 'opendss_hierarchical':
        print(f"Exporting {n_samples} OpenDSS networks to: {output_path}")
        graph = pp.topology.create_nxgraph(base_net)
        _save_to_opendss_hierarchical(base_net, graph, df_power, df_imp, output_path, n_samples)
        print(f"\nHierarchical OpenDSS export complete. {n_samples} folders created.")
        
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
def _save_to_opendss_outdated(net, graph, df_power, df_imp, output_folder, n_samples):
    """
    Loops through every sample and saves each one as a separate .dss file.
    """
    
    # Constants
    BASE_KV = net.bus.vn_kv.max()
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


def _save_to_opendss(net, graph, df_power, df_imp, output_folder, n_samples):
    """
    Loops through every sample and saves each one as a separate .dss file.
    Adapted for Multi-Voltage, Transformers, and Transmission Integration.
    """
    import os
    import numpy as np
    import warnings

    # --- Constants & Helpers ---
    DISTANCIAS = [0.6, 0.6, 1.2] 
    P_Q_RATIO = 0.328
    PHASE_TO_MATRIX_IDX = {'A': 0, 'B': 1, 'C': 2}
    PHASE_TO_DSS_NODE = {'A': '.1', 'B': '.2', 'C': '.3'}

    # --- 1. System-Wide Setup ---
    
    # Get all unique voltage levels for OpenDSS "Set Voltagebases"
    # We combine bus voltages and trafo voltages to be safe
    unique_voltages = set(net.bus['vn_kv'].dropna().unique())
    if not net.trafo.empty:
        unique_voltages.update(net.trafo['vn_hv_kv'].dropna().unique())
        unique_voltages.update(net.trafo['vn_lv_kv'].dropna().unique())
    

    #     # We strictly filter for standard LL voltages to avoid OpenDSS confusion
    raw_voltages = set(net.bus['vn_kv'].dropna().unique())
    if not net.trafo.empty:
        raw_voltages.update(net.trafo['vn_hv_kv'].dropna().unique())
        raw_voltages.update(net.trafo['vn_lv_kv'].dropna().unique())
    
    # Apply fix and sort descending (High to Low helps OpenDSS auto-resolve sometimes)

    def fix_voltage_ll(v_kv):
        if np.isclose(v_kv, 0.127):
            return 0.220  # 127V LN -> 220V LL
        if np.isclose(v_kv, 7.96): # Just in case 13.8 LN is passed
            return 13.8
        return v_kv
    # IMPORTANT: Do not include LN voltages (0.127, 7.96) in this list!
    voltage_bases_set = {fix_voltage_ll(v) for v in raw_voltages}
    voltage_bases_str = str(sorted(list(voltage_bases_set), reverse=True))

    # Slack Bus Name (Instruction 3)
    slack_bus_idx = net.ext_grid.bus.iloc[0]
    raw_slack_name = net.bus.at[slack_bus_idx, 'name']
    
    # Note: We still prefix with "bus_" for OpenDSS consistency, 
    # but the ID comes strictly from the 'name' column as requested.
    SLACK_BUS_DSS_NAME = f"bus_{raw_slack_name}"
    
    # Slack Voltage
    SLACK_KV = net.bus.vn_kv.max()

    # --- Pre-Process Indices ---
    # Ensure bus names are strings
    if 'name' not in net.bus.columns or net.bus['name'].isnull().any():
        net.bus['name'] = net.bus.index.astype(str)
    else:
        net.bus['name'] = net.bus['name'].astype(str)

    # Group data for faster lookup
    power_grouped = df_power.groupby('sample_id')
    imp_grouped = df_imp.groupby('sample_id')

    # --- Loop Samples ---
    for s in range(n_samples):
        print(f"   ... generating opendss sample {s} of {n_samples-1}", end="\r")
        
        # Get Bayesian Data for this sample
        try:
            sample_power = power_grouped.get_group(s).set_index('bus_id')
            sample_imp = imp_grouped.get_group(s).set_index('line_id')
        except KeyError:
            warnings.warn(f"Sample {s} missing in dataframe. Skipping.")
            continue
        
        dss_commands = []
        dss_commands.append("clear")
        
        # Define Circuit (Source)
        dss_commands.append(f"new circuit.synthetic_net basekv={SLACK_KV} pu=1.0 phases=3")
        # dss_commands.append(f"new circuit.synthetic_net basekv=0.22 pu=1.0 phases=3")
        dss_commands.append(f"edit Vsource.source bus1={SLACK_BUS_DSS_NAME}.1.2.3")

        # -----------------------------------------------------------
        # 2. TRANSFORMERS (New Section)
        # -----------------------------------------------------------
        if not net.trafo.empty:
            dss_commands.append("\n!---------------- TRANSFORMERS ----------------\n")
            for t_id, trafo in net.trafo.iterrows():
                
                trafo_name = trafo['name']
                if('Gen' not in trafo_name):
                    # Helper to get safe name
                    hv_bus_name = str(net.bus.at[trafo['hv_bus'], 'name'])
                    lv_bus_name = str(net.bus.at[trafo['lv_bus'], 'name'])
                    
                    # Parameters
                    # OpenDSS expects kVA, Pandapower uses MVA
                    kva = trafo['sn_mva'] * 1000
                    kv_hv = trafo['vn_hv_kv']
                    kv_lv = trafo['vn_lv_kv']
                    
                    # Impedance conversion
                    # Xhl_percent = sqrt(vk^2 - vkr^2)
                    uk = trafo['vk_percent']
                    ur = trafo['vkr_percent']
                    if uk >= ur:
                        xhl = np.sqrt(uk**2 - ur**2)
                    else:
                        xhl = 0.1 # Fallback safety
                    
                    # # Assuming 3-phase transformers for grid integration
                    dss_commands.append(
                        f"new transformer.Trafo_{t_id} phases=3 windings=2 "
                        f"buses=[bus_{hv_bus_name}.1.2.3, bus_{lv_bus_name}.1.2.3.0] "
                        f"kvas=[{kva}, {kva}] "
                        f"kvs=[{kv_hv}, {kv_lv}] "
                        f"Xhl={xhl} %loadloss={ur}"
                    )


                    # FIX: Delta-Wye Connection to handle unbalanced loads
                    # HV = Delta (3 wires), LV = Wye Grounded (4 wires)
                    # dss_commands.append(
                    #     f"new transformer.Trafo_{t_id} phases=3 windings=2 "
                    #     f"buses=[bus_{hv_bus_name}.1.2.3, bus_{lv_bus_name}.1.2.3.0] " # Note the .0 on LV
                    #     f"Conns=[Delta, Wye] " # Critical Fix
                    #     f"kvas=[{kva}, {kva}] "
                    #     f"kvs=[{kv_hv}, {kv_lv}] "
                    #     f"Xhl={xhl} %loadloss={ur}"
                    # )

        # -----------------------------------------------------------
        # 3. LINES (Adapted for Transmission/Missing Data)
        # -----------------------------------------------------------
        dss_commands.append("\n!---------------- LINES ----------------\n")
        
        for line_id, line in net.line.iterrows():
            from_bus_pp = line['from_bus']
            to_bus_pp = line['to_bus']
            
            from_bus_graph = str(net.bus.at[from_bus_pp, 'name'])
            to_bus_graph = str(net.bus.at[to_bus_pp, 'name'])

            # --- A. Determine Phases ---
            # Logic: If bus is in Bayesian samples, use sampled phase. 
            # Else (Transmission), assume 3-phase ('A','B','C').
            
            default_phases = {'A', 'B', 'C'}
            
            if from_bus_pp in sample_power.index:
                from_phases = set(sample_power.at[from_bus_pp, 'phase'])
            else:
                from_phases = default_phases
                
            if to_bus_pp in sample_power.index:
                to_phases = set(sample_power.at[to_bus_pp, 'phase'])
            else:
                to_phases = default_phases

            line_phases_set = from_phases.intersection(to_phases)

            if not line_phases_set:
                # Disconnected lines (rare but possible in graph cleaning)
                continue

            line_phases_sorted = sorted(list(line_phases_set))
            num_phases = len(line_phases_sorted)

            # --- B. Get Impedance ---
            # Logic: If line is in Bayesian samples, use generated R1/X1.
            # Else (Transmission), use static R/X from net.line columns.
            
            if line_id in sample_imp.index:
                r1 = sample_imp.at[line_id, 'R1_ohm_per_km']
                x1 = sample_imp.at[line_id, 'X1_ohm_per_km']
            else:
                # Transmission/Static Line
                r1 = line['r_ohm_per_km']
                x1 = line['x_ohm_per_km']

            # Sanity floor to prevent division by zero in matrices
            r1 = max(r1, 1e-6)
            x1 = max(x1, 1e-6)

            #print(r1)
            
            # --- C. Matrix Construction ---
            Zabc_full, _ = carson_modificada_Zabc(r1, x1, DISTANCIAS)

            # Subset Zabc matrix based on active phases
            matrix_indices = [PHASE_TO_MATRIX_IDX[p] for p in line_phases_sorted]
            Zabc_subset = Zabc_full[np.ix_(matrix_indices, matrix_indices)]
            rmatrix_subset = np.real(Zabc_subset)
            xmatrix_subset = np.imag(Zabc_subset)

            # Format for DSS
            linecode_name = f"LCode_{line_id}"
            r_str = format_matrix_for_dss(rmatrix_subset)
            x_str = format_matrix_for_dss(xmatrix_subset)
            
            dss_commands.append(
                f"new linecode.{linecode_name} nphases={num_phases} "
                f"rmatrix={r_str} xmatrix={x_str} units=km"
            )

            # --- D. Create Line Element ---
            line_name = f"line_{line_id}"
            dss_phase_nodes = "".join([PHASE_TO_DSS_NODE[p] for p in line_phases_sorted])
            
            from_bus_dss = f"bus_{from_bus_graph}{dss_phase_nodes}"
            to_bus_dss = f"bus_{to_bus_graph}{dss_phase_nodes}"
            line_len_km = line['length_km']

            dss_commands.append(
                f"new line.{line_name} bus1={from_bus_dss} bus2={to_bus_dss} "
                f"linecode={linecode_name} length={line_len_km} units=km"
            )
        
        # -----------------------------------------------------------
        # 4. LOADS (Adapted for Bus-Specific Voltage)
        # -----------------------------------------------------------
        dss_commands.append("\n!---------------- LOADS ----------------\n")
        
        for bus_id, bus in net.bus.iterrows():
            bus_graph_id = str(bus['name'])
            
            # Skip if this is the slack bus
            if f"bus_{bus_graph_id}" == SLACK_BUS_DSS_NAME:
                continue
                
            # Skip if bus not in power data (Transmission bus with no load)
            if bus_id not in sample_power.index:
                continue
                
            bus_data = sample_power.loc[bus_id]
            bus_phases = bus_data['phase']
            
            # Important: Use the specific bus voltage level
            BUS_KV = bus['vn_kv']
            
            # Write load for each phase present
            BUS_KV = BUS_KV/np.sqrt(3)
            #print(BUS_KV)
            if(BUS_KV < 0.5): # only low voltage buses have loads 
                for p in ['A', 'B', 'C']:
                    col = f'P_{p}'

                    if p in bus_phases and bus_data[col] > 0:
                        kw = bus_data[col]
                        kvar = kw * P_Q_RATIO
                        
                        # OpenDSS node suffix (.1, .2, .3)
                        node = PHASE_TO_DSS_NODE[p]


                        
                        dss_commands.append(
                            f"new load.load_{bus_graph_id}_{p} "
                            f"bus1=bus_{bus_graph_id}{node} "
                            f"Phases=1 kv={BUS_KV} " # Correct voltage
                            f"kw={kw/1000} kvar={kvar/1000} "
                            f"vminpu=0.85 vmaxpu=1.15 model=1"
                        )

        # -----------------------------------------------------------
        # 5. SOLUTION & VOLTAGE BASES
        # -----------------------------------------------------------
        dss_commands.append("\n!---------------- SOLUTION ----------------\n")
        dss_commands.append("set controlmode=off")
        # Set all voltage bases found in the system
        dss_commands.append(f"Set Voltagebases={voltage_bases_str}")
        dss_commands.append("calcv")
        dss_commands.append("Solve")
        
        # Write File
        file_path = os.path.join(output_folder, f"master_sample_{s}.dss")
        with open(file_path, 'w') as f:
            f.write("\n".join(dss_commands))
            
    print("\n")

def _save_to_opendss(net, graph, df_power, df_imp, output_folder, n_samples):
    import os
    import numpy as np
    import warnings

    # --- Constants ---
    DISTANCIAS = [0.6, 0.6, 1.2] 
    P_Q_RATIO = 0.328
    PHASE_TO_MATRIX_IDX = {'A': 0, 'B': 1, 'C': 2}
    PHASE_TO_DSS_NODE = {'A': '.1', 'B': '.2', 'C': '.3'}

    # Helper: Convert 127V (LN) to 220V (LL) for 3-phase definitions
    def fix_voltage_ll(v_kv):
        if np.isclose(v_kv, 0.127):
            return 0.220  # 127V LN -> 220V LL
        if np.isclose(v_kv, 7.96): # Just in case 13.8 LN is passed
            return 13.8
        return v_kv

    # --- 1. System Setup ---
    # Collect unique Line-to-Line voltages for Set Voltagebases
    # We strictly filter for standard LL voltages to avoid OpenDSS confusion
    raw_voltages = set(net.bus['vn_kv'].dropna().unique())
    if not net.trafo.empty:
        raw_voltages.update(net.trafo['vn_hv_kv'].dropna().unique())
        raw_voltages.update(net.trafo['vn_lv_kv'].dropna().unique())
    
    # Apply fix and sort descending (High to Low helps OpenDSS auto-resolve sometimes)
    # IMPORTANT: Do not include LN voltages (0.127, 7.96) in this list!
    voltage_bases_set = {fix_voltage_ll(v) for v in raw_voltages}
    voltage_bases_str = str(sorted(list(voltage_bases_set), reverse=True))

    # Slack Bus
    slack_bus_idx = net.ext_grid.bus.iloc[0]
    raw_slack_name = net.bus.at[slack_bus_idx, 'name']
    SLACK_BUS_DSS_NAME = f"bus_{raw_slack_name}"
    
    # Ensure Slack KV is Line-to-Line (e.g. 13.8 not 7.96)
    SLACK_KV = fix_voltage_ll(net.bus.at[slack_bus_idx, 'vn_kv'])

    net.bus['name'] = net.bus['name'].astype(str)
    power_grouped = df_power.groupby('sample_id')
    imp_grouped = df_imp.groupby('sample_id')

    # --- Loop Samples ---
    for s in range(n_samples):
        print(f"   ... generating opendss sample {s} of {n_samples}", end="\r")
        
        try:
            sample_power = power_grouped.get_group(s).set_index('bus_id')
            sample_imp = imp_grouped.get_group(s).set_index('line_id')
        except KeyError:
            continue
        
        dss_commands = []
        dss_commands.append("clear")
        
        # 1. Circuit
        dss_commands.append(f"new circuit.synthetic_net basekv={SLACK_KV} pu=1.0 phases=3")
        # Ensure source is stiff
        # (OLD) dss_commands.append(f"edit Vsource.source bus1={SLACK_BUS_DSS_NAME}.1.2.3")
        # 1. Handle the first ext_grid as the default 'Vsource.source'
        slack_bus_idx = net.ext_grid.bus.iloc[0]
        raw_slack_name = net.bus.at[slack_bus_idx, 'name']
        SLACK_BUS_DSS_NAME = f"bus_{raw_slack_name}"

        # Ensure Slack KV is Line-to-Line
        SLACK_KV = fix_voltage_ll(net.bus.at[slack_bus_idx, 'vn_kv'])

        # Edit the default source
        dss_commands.append(f"edit Vsource.source bus1={SLACK_BUS_DSS_NAME}.1.2.3 basekv={SLACK_KV} pu=1.0 phases=3 MVAsc3=2000 MVAsc1=2000")

        # 2. Iterate through all OTHER ext_grids to create additional sources
        if len(net.ext_grid) > 1:
            for i in range(1, len(net.ext_grid)):
                other_slack_idx = net.ext_grid.bus.iloc[i]
                other_raw_name = net.bus.at[other_slack_idx, 'name']
                OTHER_BUS_NAME = f"bus_{other_raw_name}"
                OTHER_KV = fix_voltage_ll(net.bus.at[other_slack_idx, 'vn_kv'])
                
                # Add a new Vsource for each island head
                # We use i as a suffix to ensure unique names (source1, source2, etc.)
                dss_commands.append(
                    f"New Vsource.source{i} bus1={OTHER_BUS_NAME}.1.2.3 "
                    f"basekv={OTHER_KV} phases=3 pu=1.0 MVAsc3=2000 MVAsc1=2000"
                )
        # # 2. (OLD) Transformers (Delta-Wye Fix)
        # if not net.trafo.empty:
        #     dss_commands.append("\n!---------------- TRANSFORMERS ----------------\n")
        #     for t_id, trafo in net.trafo.iterrows():
        #         hv_name = str(net.bus.at[trafo['hv_bus'], 'name'])
        #         lv_name = str(net.bus.at[trafo['lv_bus'], 'name'])
                
        #         kva = trafo['sn_mva'] * 1000.0
        #         kv_hv = fix_voltage_ll(trafo['vn_hv_kv'])
        #         kv_lv = fix_voltage_ll(trafo['vn_lv_kv'])
                
        #         uk, ur = trafo['vk_percent'], trafo['vkr_percent']
        #         xhl = np.sqrt(uk**2 - ur**2) if uk >= ur else 0.1
                
        #         # FIX: Delta-Wye Connection to handle unbalanced loads
        #         # HV = Delta (3 wires), LV = Wye Grounded (4 wires)
        #         dss_commands.append(
        #             f"new transformer.Trafo_{t_id} phases=3 windings=2 "
        #             f"buses=[bus_{hv_name}.1.2.3, bus_{lv_name}.1.2.3.0] " # Note the .0 on LV
        #             f"Conns=[Delta, Wye] " # Critical Fix
        #             f"kvas=[{kva}, {kva}] "
        #             f"kvs=[{kv_hv}, {kv_lv}] "
        #             f"Xhl={xhl} %loadloss={ur}"
        #         )



        # 2. Transformers (Delta-Wye Fix + Phase Lookup from sample_power)
        if not net.trafo.empty:
            print('novo novo')
            dss_commands.append("\n!---------------- TRANSFORMERS ----------------\n")
            
            # Map letters to OpenDSS node numbers
            phase_map = {'A': '1', 'B': '2', 'C': '3'}
            default_phases = {'A', 'B', 'C'}

            for t_id, trafo in net.trafo.iterrows():
                hv_bus_id = trafo['hv_bus']
                lv_bus_id = trafo['lv_bus']
                
                hv_name = str(net.bus.at[hv_bus_id, 'name'])
                lv_name = str(net.bus.at[lv_bus_id, 'name'])
                
                # --- LOGIC UPDATE: Get Phase from sample_power ---
                # We check the HV bus in sample_power to see which phases exist there
                if hv_bus_id in sample_power.index:
                    # Retrieve phase info (e.g., 'A', 'B', 'C', or a set like {'A','B'})
                    raw_phase = sample_power.at[hv_bus_id, 'phase']
                    # Ensure it is treated as a set for consistency
                    current_phases = set(raw_phase)
                else:
                    current_phases = default_phases

                # Determine if we are Single-Phase or Three-Phase
                num_phases = len(current_phases)

                # Standard variables
                kva = trafo['sn_mva'] * 1000.0
                kv_hv = fix_voltage_ll(trafo['vn_hv_kv'])
                kv_lv = fix_voltage_ll(trafo['vn_lv_kv'])
                
                uk, ur = trafo['vk_percent'], trafo['vkr_percent']
                xhl = np.sqrt(uk**2 - ur**2) if uk >= ur else 0.1

                # --- CONFIGURATION LOGIC ---
                if num_phases == 1:
                    # >> Single Phase Configuration <<
                    # 1. Topology: Wye-Wye (Line-Neutral) prevents floating delta issues
                    conn_type = "[Wye, Wye]"
                    
                    # 2. Voltage: Convert Line-Line rating to Line-Neutral
                    kv_hv = kv_hv / np.sqrt(3)
                    kv_lv = kv_lv / np.sqrt(3)
                    
                    # 3. Nodal Connection: Map 'A'->'.1', etc.
                    # Pop the single phase letter from the set
                    phase_char = list(current_phases)[0]
                    dss_node = phase_map.get(phase_char, '1') # Default to 1 if unknown char
                    
                    # Construct bus strings (e.g., bus_X.3, bus_Y.3.0)
                    hv_bus_str = f"bus_{hv_name}.{dss_node}"
                    lv_bus_str = f"bus_{lv_name}.{dss_node}.0"
                    
                    phases_param = 1
                    
                else:
                    # >> Three Phase Configuration <<
                    # 1. Topology: Delta-Wye (Standard Distribution)
                    conn_type = "[Delta, Wye]"
                    
                    # 2. Voltage: Remains Line-to-Line (no change)
                    
                    # 3. Nodal Connection: Connect all 3 phases
                    hv_bus_str = f"bus_{hv_name}.1.2.3"
                    lv_bus_str = f"bus_{lv_name}.1.2.3.0"
                    
                    phases_param = 3

                # conn_type = "[Delta, Wye]"
                
                # # 2. Voltage: Remains Line-to-Line (no change)
                
                # # 3. Nodal Connection: Connect all 3 phases
                # hv_bus_str = f"bus_{hv_name}.1.2.3"
                # lv_bus_str = f"bus_{lv_name}.1.2.3.0"
                
                # phases_param = 3

                # Generate OpenDSS Command
                dss_commands.append(
                    f"new transformer.Trafo_{t_id} phases={phases_param} windings=2 "
                    f"buses=[{hv_bus_str}, {lv_bus_str}] "
                    f"Conns={conn_type} "
                    f"kvas=[{kva}, {kva}] "
                    f"kvs=[{kv_hv:.4f}, {kv_lv:.4f}] "
                    f"Xhl={xhl} %loadloss={ur}"
                )

        # 3. Lines
        dss_commands.append("\n!---------------- LINES ----------------\n")
        for line_id, line in net.line.iterrows():
            from_pp, to_pp = line['from_bus'], line['to_bus']
            from_name = str(net.bus.at[from_pp, 'name'])
            to_name = str(net.bus.at[to_pp, 'name'])

            default = {'A','B','C'}
            p_from = set(sample_power.at[from_pp, 'phase']) if from_pp in sample_power.index else default
            p_to = set(sample_power.at[to_pp, 'phase']) if to_pp in sample_power.index else default
            
            common = sorted(list(p_from.intersection(p_to)))

            if(len(common) == 2): # we dont have 2 phase transformers so we must use line codes with 3 phases and just connect the 2 phases we have.
                common = ['A', 'B', 'C']
            
            if not common: continue

            if line_id in sample_imp.index:
                r1 = sample_imp.at[line_id, 'R1_ohm_per_km']
                x1 = sample_imp.at[line_id, 'X1_ohm_per_km']
            else:
                r1, x1 = line['r_ohm_per_km'], line['x_ohm_per_km']
            
            r1, x1 = max(r1, 1e-6), max(x1, 1e-6)
            Zabc_full, _ = carson_modificada_Zabc(r1, x1, DISTANCIAS)
            
            idx = [PHASE_TO_MATRIX_IDX[p] for p in common]
            Z_sub = Zabc_full[np.ix_(idx, idx)]
            
            lcode = f"LCode_{line_id}"
            r_str = format_matrix_for_dss(np.real(Z_sub))
            x_str = format_matrix_for_dss(np.imag(Z_sub))
            
            dss_commands.append(f"new linecode.{lcode} nphases={len(common)} rmatrix={r_str} xmatrix={x_str} units=km")
            
            nodes = "".join([PHASE_TO_DSS_NODE[p] for p in common])
            dss_commands.append(
                f"new line.line_{line_id} bus1=bus_{from_name}{nodes} bus2=bus_{to_name}{nodes} "
                f"linecode={lcode} length={0.1*line['length_km']} units=km"
            )

        # 4. Loads
        dss_commands.append("\n!---------------- LOADS ----------------\n")
        for bus_id, bus in net.bus.iterrows():
            b_name = str(bus['name'])
            if f"bus_{b_name}" == SLACK_BUS_DSS_NAME: continue
            if bus_id not in sample_power.index: continue
            if bus['vn_kv'] > 0.5: continue # Only add loads to low voltage buses 
            
            b_data = sample_power.loc[bus_id]
            
            # Load voltage: If on LV (0.127), we use 0.127 (LN). 
            # If on MV (13.8), we use 7.96 (LN).
            bus_kv_ll = fix_voltage_ll(bus['vn_kv'])
            #load_kv_ln = bus_kv_ll / np.sqrt(3) 
            load_kv_ln = bus_kv_ll   

            # Special case: If the input data already said 0.127, keep it exact.
            if np.isclose(bus['vn_kv'], 0.127):
                load_kv_ln = 0.127

            for p in ['A', 'B', 'C']:
                col = f'P_{p}'
                if p in b_data['phase'] and b_data[col] > 0:
                    kw = b_data[col] # User confirmed this is kW
                    dss_commands.append(
                        f"new load.load_{b_name}_{p} bus1=bus_{b_name}{PHASE_TO_DSS_NODE[p]} "
                        f"Phases=1 kv={load_kv_ln:.4f} kw={kw/1000} kvar={kw*P_Q_RATIO/1000} "
                        f"vminpu=0.85 vmaxpu=1.15 model=1"
                    )

        # 5. Solution
        dss_commands.append("\n!---------------- SOLUTION ----------------\n")
        dss_commands.append("set controlmode=off")
        # Ensure ONLY Line-to-Line voltages are here
        dss_commands.append(f"Set Voltagebases={voltage_bases_str}")
        dss_commands.append("calcv")
        dss_commands.append("Solve")
        
        with open(os.path.join(output_folder, f"master_sample_{s}.dss"), 'w') as f:
            f.write("\n".join(dss_commands))
            
    print("\n")


def _save_to_opendss_hierarchical(net, graph, df_power, df_imp, output_folder, n_samples):
    import os
    import numpy as np
    import networkx as nx
    import pandas as pd
    import shutil

    # --- Constants ---
    DISTANCIAS = [0.6, 0.6, 1.2] 
    P_Q_RATIO = 0.328
    PHASE_TO_MATRIX_IDX = {'A': 0, 'B': 1, 'C': 2}
    PHASE_TO_DSS_NODE = {'A': '.1', 'B': '.2', 'C': '.3'}
    PHASE_MAP = {'A': '1', 'B': '2', 'C': '3'}

    # Helper: Convert 127V (LN) to 220V (LL) for 3-phase definitions
    def fix_voltage_ll(v_kv):
        if np.isclose(v_kv, 0.127):
            return 0.220  
        if np.isclose(v_kv, 7.96): 
            return 13.8
        return v_kv

    # --- 1. System Analysis ---
    # Identify unique voltages for BaseKV
    raw_voltages = set(net.bus['vn_kv'].dropna().unique())
    if not net.trafo.empty:
        raw_voltages.update(net.trafo['vn_hv_kv'].dropna().unique())
        raw_voltages.update(net.trafo['vn_lv_kv'].dropna().unique())
    
    voltage_bases_set = {fix_voltage_ll(v) for v in raw_voltages}
    if any(np.isclose(v, 0.127) for v in raw_voltages):
        voltage_bases_set.add(0.127)
    
    voltage_bases_str = str(sorted(list(voltage_bases_set), reverse=True))

    # Slack Bus Info
    slack_bus_idx = net.ext_grid.bus.iloc[0]
    slack_bus_idx_list = net.ext_grid.bus
    SLACK_KV = fix_voltage_ll(net.bus.at[slack_bus_idx, 'vn_kv'])

    net.bus['name'] = net.bus['name'].astype(str)
    power_grouped = df_power.groupby('sample_id')
    imp_grouped = df_imp.groupby('sample_id')

    # --- 2. Topology Splitting (Islands) ---
    print("   ... analyzing network topology for islands ...")
    g_net = nx.Graph()
    g_net.add_nodes_from(net.bus.index)
    for _, line in net.line.iterrows():
        g_net.add_edge(line['from_bus'], line['to_bus'], id=_)

    islands = list(nx.connected_components(g_net))
    
    mv_island_nodes = set()
    lv_islands_map = {} 

    # Identify MV Island (Has Ext Grid)
    for island in islands:
        if any(bus in island for bus in slack_bus_idx_list):
            mv_island_nodes = island
            break
    
    # Map remaining islands to their source Transformers
    for t_id, trafo in net.trafo.iterrows():
        lv_bus = trafo['lv_bus']
        for island in islands:
            if lv_bus in island:
                lv_islands_map[t_id] = island
                break

    # --- Loop Samples ---
    for s in range(n_samples):
        print(f"   ... generating opendss folder for sample {s} of {n_samples}", end="\r")
        
        try:
            sample_power = power_grouped.get_group(s).set_index('bus_id')
            sample_imp = imp_grouped.get_group(s).set_index('line_id')
        except KeyError:
            continue

        # Create Root Folder
        sample_folder = os.path.join(output_folder, f"Sample_{s}")
        if os.path.exists(sample_folder): shutil.rmtree(sample_folder)
        os.makedirs(sample_folder)
        
        master_redirects = []

        # ==========================================
        # A. MV Network (Root Folder Files)
        # ==========================================
        
        # A1. VSource.dss
        vsource_cmds = []
        vsource_cmds.append(f"new circuit.synthetic_net basekv={SLACK_KV} pu=1.0 phases=3")
        for ext_grid_idx in net.ext_grid.bus:
            vsource_cmds.append(f"edit Vsource.source bus1=bus_{net.bus.at[ext_grid_idx, 'name']}.1.2.3 basekv={fix_voltage_ll(net.bus.at[ext_grid_idx, 'vn_kv'])} pu=1.0 phases=3 MVAsc3=2000 MVAsc1=2000")
        
        if len(net.ext_grid) > 1:
            for i in range(1, len(net.ext_grid)):
                other_idx = net.ext_grid.bus.iloc[i]
                other_name = net.bus.at[other_idx, 'name']
                other_kv = fix_voltage_ll(net.bus.at[other_idx, 'vn_kv'])
                vsource_cmds.append(f"New Vsource.source{i} bus1=bus_{other_name}.1.2.3 basekv={other_kv} phases=3 pu=1.0 MVAsc3=2000 MVAsc1=2000")

        with open(os.path.join(sample_folder, "VSource.dss"), 'w') as f:
            f.write("\n".join(vsource_cmds))
        # master_redirects.append("VSource.dss")

        # A2. LineCodes.dss (Global)
        linecode_cmds = []
        for line_id, line in net.line.iterrows():
            from_pp, to_pp = line['from_bus'], line['to_bus']
            from_name = str(net.bus.at[from_pp, 'name'])
            to_name = str(net.bus.at[to_pp, 'name'])

            default = {'A','B','C'}
            p_from = set(sample_power.at[from_pp, 'phase']) if from_pp in sample_power.index else default
            p_to = set(sample_power.at[to_pp, 'phase']) if to_pp in sample_power.index else default
            
            common = sorted(list(p_from.intersection(p_to)))

            if(len(common) == 2): # we dont have 2 phase transformers so we must use line codes with 3 phases and just connect the 2 phases we have.
                common = ['A', 'B', 'C']


            if line_id in sample_imp.index:
                r1 = sample_imp.at[line_id, 'R1_ohm_per_km']
                x1 = sample_imp.at[line_id, 'X1_ohm_per_km']
            else:
                r1, x1 = line['r_ohm_per_km'], line['x_ohm_per_km']
            
            r1, x1 = max(r1, 1e-6), max(x1, 1e-6)
            Zabc_full, _ = carson_modificada_Zabc(r1, x1, DISTANCIAS)
            
            idx = [PHASE_TO_MATRIX_IDX[p] for p in common]
            Z_sub = Zabc_full[np.ix_(idx, idx)]
            
            lcode = f"LCode_{line_id}"
            r_str = format_matrix_for_dss(np.real(Z_sub))
            x_str = format_matrix_for_dss(np.imag(Z_sub))
            
            linecode_cmds.append(f"new linecode.{lcode} nphases={len(common)} rmatrix={r_str} xmatrix={x_str} units=km")
            

        with open(os.path.join(sample_folder, "LineCodes.dss"), 'w') as f:
            f.write("\n".join(linecode_cmds))
        # master_redirects.append("LineCodes.dss")

        # A3. MV_Lines.dss
        mv_lines_cmds = []
        mv_lines_cmds.append("! MV Feeder Lines")
        for line_id, line in net.line.iterrows():
            if line['from_bus'] in mv_island_nodes and line['to_bus'] in mv_island_nodes:
                from_name = net.bus.at[line['from_bus'], 'name']
                to_name = net.bus.at[line['to_bus'], 'name']
                
                p_from = set(sample_power.at[line['from_bus'], 'phase']) if line['from_bus'] in sample_power.index else {'A','B','C'}
                p_to = set(sample_power.at[line['to_bus'], 'phase']) if line['to_bus'] in sample_power.index else {'A','B','C'}
                common = sorted(list(p_from.intersection(p_to)))
                if not common: continue
                if len(common) == 2: common = ['A', 'B', 'C']
                
                nodes = "".join([PHASE_TO_DSS_NODE[p] for p in common])
                
                mv_lines_cmds.append(
                    f"new line.line_{line_id} bus1=bus_{from_name}{nodes} bus2=bus_{to_name}{nodes} "
                    f"linecode=LCode_{line_id} length={0.1*line['length_km']} units=km"
                )

        with open(os.path.join(sample_folder, "MV_Lines.dss"), 'w') as f:
            f.write("\n".join(mv_lines_cmds))
        # master_redirects.append("MV_Lines.dss")

        # ==========================================
        # B. LV Networks (Subfolders)
        # ==========================================
        
        for t_id, trafo in net.trafo.iterrows():
            if t_id not in lv_islands_map: continue
            
            island_nodes = lv_islands_map[t_id]
            lv_subfolder_name = f"LV_Net_Trafo_{t_id}"
            lv_path = os.path.join(sample_folder, lv_subfolder_name)
            os.makedirs(lv_path)
            
            # --- B1. Transformer.dss ---
            trafo_cmds = []
            hv_bus_id = trafo['hv_bus']
            lv_bus_id = trafo['lv_bus']
            hv_name = net.bus.at[hv_bus_id, 'name']
            lv_name = net.bus.at[lv_bus_id, 'name']
            
            if hv_bus_id in sample_power.index:
                current_phases = set(sample_power.at[hv_bus_id, 'phase'])
            else:
                current_phases = {'A', 'B', 'C'}
            num_phases = len(current_phases)

            kva = trafo['sn_mva'] * 1000.0
            kv_hv = fix_voltage_ll(trafo['vn_hv_kv'])
            kv_lv = fix_voltage_ll(trafo['vn_lv_kv'])
            uk, ur = trafo['vk_percent'], trafo['vkr_percent']
            xhl = np.sqrt(uk**2 - ur**2) if uk >= ur else 0.1

            # Determine nodes and connection type for both Transformer AND Local Master
            if num_phases == 1:
                # Single Phase Logic (Wye-Wye grounded)
                conn_type = "[Wye, Wye]"
                kv_hv_dss = kv_hv / np.sqrt(3) # L-N
                kv_lv_dss = kv_lv / np.sqrt(3) # L-N
                phase_char = list(current_phases)[0]
                dss_node = PHASE_MAP.get(phase_char, '1')
                hv_bus_str = f"bus_{hv_name}.{dss_node}"
                lv_bus_str = f"bus_{lv_name}.{dss_node}.0"
                phases_param = 1
            else:
                # Three Phase Logic (Delta-Wye)
                conn_type = "[Delta, Wye]"
                kv_hv_dss = kv_hv # L-L
                kv_lv_dss = kv_lv # L-L
                hv_bus_str = f"bus_{hv_name}.1.2.3"
                lv_bus_str = f"bus_{lv_name}.1.2.3.0"
                phases_param = 3

            trafo_cmds.append(
                f"new transformer.Trafo_{t_id} phases={phases_param} windings=2 "
                f"buses=[{hv_bus_str}, {lv_bus_str}] Conns={conn_type} "
                f"kvas=[{kva}, {kva}] kvs=[{kv_hv_dss:.4f}, {kv_lv_dss:.4f}] "
                f"Xhl={xhl} %loadloss={ur}"
            )
            
            with open(os.path.join(lv_path, "Transformer.dss"), 'w') as f:
                f.write("\n".join(trafo_cmds))

            # --- B2. Lines.dss (LV) ---
            lv_lines_cmds = []
            for line_id, line in net.line.iterrows():
                if line['from_bus'] in island_nodes and line['to_bus'] in island_nodes:
                    from_name = net.bus.at[line['from_bus'], 'name']
                    to_name = net.bus.at[line['to_bus'], 'name']
                    
                    p_from = set(sample_power.at[line['from_bus'], 'phase']) if line['from_bus'] in sample_power.index else {'A','B','C'}
                    nodes = "".join([PHASE_TO_DSS_NODE[p] for p in sorted(list(p_from))])
                    
                    lv_lines_cmds.append(
                        f"new line.line_{line_id} bus1=bus_{from_name}{nodes} bus2=bus_{to_name}{nodes} "
                        f"linecode=LCode_{line_id} length={0.1*line['length_km']} units=km"
                    )
            with open(os.path.join(lv_path, "Lines.dss"), 'w') as f:
                f.write("\n".join(lv_lines_cmds))

            # --- B3. Loads.dss (LV) ---
            load_cmds = []
            for bus_id in island_nodes:
                if bus_id not in sample_power.index: continue
                bus_data = net.bus.loc[bus_id]
                if bus_data['vn_kv'] > 0.5: continue 

                b_name = bus_data['name']
                b_power = sample_power.loc[bus_id]
                
                load_kv = fix_voltage_ll(bus_data['vn_kv'])
                if np.isclose(bus_data['vn_kv'], 0.127): load_kv = 0.127

                for p in ['A', 'B', 'C']:
                    col = f'P_{p}'
                    if p in b_power['phase'] and b_power[col] > 0:
                        kw = b_power[col]
                        load_cmds.append(
                            f"new load.load_{b_name}_{p} bus1=bus_{b_name}{PHASE_TO_DSS_NODE[p]} "
                            f"Phases=1 kv={load_kv:.4f} kw={kw/1000} kvar={kw*P_Q_RATIO/1000} "
                            f"vminpu=0.85 vmaxpu=1.15 model=1"
                        )
            with open(os.path.join(lv_path, "Loads.dss"), 'w') as f:
                f.write("\n".join(load_cmds))

            # --- B4. Master_Local.dss (The Standalone Simulation) ---
            # This file mimics the transformer output using a VSource, allowing independent LV sim.
            local_master_cmds = []
            local_master_cmds.append("Clear")
            
            # Define Circuit and Source at the LV bus (Transformer replacement)
            local_master_cmds.append(f"new circuit.LV_Island_{t_id} basekv={kv_lv_dss:.4f} pu=1.0 phases={phases_param}")
            
            # Note: We use the exact lv_bus_str derived in the Transformer logic (e.g. bus_Name.1.2.3.0 or bus_Name.1.0)
            # This ensures the source is grounded exactly how the transformer wye point was.
            local_master_cmds.append(
                f"edit Vsource.source bus1={lv_bus_str} basekv={kv_lv_dss:.4f} pu=1.0 phases={phases_param} MVAsc3=1000 MVAsc1=1000"
            )

            # Redirects (Note relative path for LineCodes)
            local_master_cmds.append("Redirect ../LineCodes.dss") 
            local_master_cmds.append("Redirect Lines.dss")
            local_master_cmds.append("Redirect Loads.dss")
            
            local_master_cmds.append("set controlmode=off")
            local_master_cmds.append(f"Set Voltagebases=[{kv_lv_dss:.4f}]")
            local_master_cmds.append("calcv")
            local_master_cmds.append("Solve")

            with open(os.path.join(lv_path, "Master_Local.dss"), 'w') as f:
                f.write("\n".join(local_master_cmds))

            # Add redirects to Global Master
            master_redirects.append(f"Redirect {lv_subfolder_name}/Transformer.dss")
            master_redirects.append(f"Redirect {lv_subfolder_name}/Lines.dss")
            master_redirects.append(f"Redirect {lv_subfolder_name}/Loads.dss")

        # ==========================================
        # C. Master.dss (Global)
        # ==========================================
        master_cmds = []
        master_cmds.append("Clear")
        # 1. Global Definitions
        master_redirects.insert(0, "Redirect MV_Lines.dss") # MV lines come before Trafo redirects usually
        master_redirects.insert(0, "Redirect LineCodes.dss")
        master_redirects.insert(0, "Redirect VSource.dss")
        
        for cmd in master_redirects:
            master_cmds.append(f"{cmd}")
        
        master_cmds.append("set controlmode=off")
        master_cmds.append(f"Set Voltagebases={voltage_bases_str}")
        master_cmds.append("calcv")
        master_cmds.append("Solve")

        with open(os.path.join(sample_folder, "Master.dss"), 'w') as f:
            f.write("\n".join(master_cmds))
            
    print("\nGeneration Complete.")


def _save_to_opendss_hierarchical(net, graph, df_power, df_imp, output_folder, n_samples):
    import os
    import numpy as np
    import networkx as nx
    import pandas as pd
    import shutil

    # --- Constants ---
    DISTANCIAS = [0.6, 0.6, 1.2] 
    P_Q_RATIO = 0.328
    PHASE_TO_MATRIX_IDX = {'A': 0, 'B': 1, 'C': 2}
    PHASE_TO_DSS_NODE = {'A': '.1', 'B': '.2', 'C': '.3'}
    PHASE_MAP = {'A': '1', 'B': '2', 'C': '3'}

    def fix_voltage_ll(v_kv):
        if np.isclose(v_kv, 0.127): return 0.220  
        if np.isclose(v_kv, 7.96): return 13.8
        return v_kv


    # --- 1. System Analysis ---
    raw_voltages = set(net.bus['vn_kv'].dropna().unique())
    if not net.trafo.empty:
        raw_voltages.update(net.trafo['vn_hv_kv'].dropna().unique())
        raw_voltages.update(net.trafo['vn_lv_kv'].dropna().unique())
    
    voltage_bases_set = {fix_voltage_ll(v) for v in raw_voltages}
    if any(np.isclose(v, 0.127) for v in raw_voltages):
        voltage_bases_set.add(0.127)
    voltage_bases_str = str(sorted(list(voltage_bases_set), reverse=True))

    # Identify all Source Buses (Ext Grids)
    source_buses = set(net.ext_grid.bus.values)
    
    # Get Primary Slack (for the 'New Circuit' command)
    slack_bus_idx = net.ext_grid.bus.iloc[0]
    SLACK_KV = fix_voltage_ll(net.bus.at[slack_bus_idx, 'vn_kv'])

    net.bus['name'] = net.bus['name'].astype(str)
    power_grouped = df_power.groupby('sample_id')
    imp_grouped = df_imp.groupby('sample_id')

    # --- 2. Topology Splitting (Multi-Source Aware) ---
    print("   ... analyzing network topology for islands ...")
    g_net = nx.Graph()
    g_net.add_nodes_from(net.bus.index)
    for _, line in net.line.iterrows():
        g_net.add_edge(line['from_bus'], line['to_bus'], id=_)

    # Get all disconnected islands
    islands = list(nx.connected_components(g_net))
    
    mv_nodes_global = set() # All nodes belonging to ANY source-fed island
    lv_islands_map = {}     # Map Trafo_ID -> Island Nodes

    # 1. Identify MV Islands (Any island containing ANY ext_grid bus)
    for island in islands:
        # If this island contains *any* source bus, it's part of the MV network
        if not source_buses.isdisjoint(island):
            mv_nodes_global.update(island)

    # 2. Identify LV Islands (Downstream of Transformers)
    for t_id, trafo in net.trafo.iterrows():
        lv_bus = trafo['lv_bus']
        # Find which island this LV bus belongs to
        for island in islands:
            if lv_bus in island:
                # Double check: An LV island should NOT contain a Source (ExtGrid)
                # If it does, it's likely the transformer is meshed back to MV (unlikely in radial)
                if source_buses.isdisjoint(island):
                    lv_islands_map[t_id] = island
                break

    # --- Loop Samples ---
    for s in range(n_samples):
        print(f"   ... generating opendss folder for sample {s} of {n_samples}", end="\r")
        try:
            sample_power = power_grouped.get_group(s).set_index('bus_id')
            sample_imp = imp_grouped.get_group(s).set_index('line_id')
        except KeyError:
            continue

        sample_folder = os.path.join(output_folder, f"Sample_{s}")
        if os.path.exists(sample_folder): shutil.rmtree(sample_folder)
        os.makedirs(sample_folder)
        
        master_redirects = []

        # ==========================================
        # A. MV Network (Root Folder)
        # ==========================================
        
        # A1. VSource.dss (Handles Multi-Source)
        vsource_cmds = []
        # Define Circuit using the FIRST source
        first_source_bus_name = net.bus.at[slack_bus_idx, 'name']
        vsource_cmds.append(f"new circuit.synthetic_net basekv={SLACK_KV} pu=1.0 phases=3")
        
        # Edit the default source created by 'new circuit'
        vsource_cmds.append(
            f"edit Vsource.source bus1=bus_{first_source_bus_name}.1.2.3 basekv={SLACK_KV} pu=1.0 phases=3 MVAsc3=2000 MVAsc1=2000"
        )
        
        # Add NEW Vsources for any additional ExtGrids (if multi-feeder)
        if len(net.ext_grid) > 1:
            for i in range(1, len(net.ext_grid)):
                other_idx = net.ext_grid.bus.iloc[i]
                other_name = net.bus.at[other_idx, 'name']
                other_kv = fix_voltage_ll(net.bus.at[other_idx, 'vn_kv'])
                # Using unique names source_1, source_2...
                vsource_cmds.append(
                    f"New Vsource.source_{i} bus1=bus_{other_name}.1.2.3 basekv={other_kv} phases=3 pu=1.0 MVAsc3=2000 MVAsc1=2000"
                )

        with open(os.path.join(sample_folder, "VSource.dss"), 'w') as f:
            f.write("\n".join(vsource_cmds))

        # A2. LineCodes.dss (Global)
        linecode_cmds = []
        for line_id, line in net.line.iterrows():
            from_pp, to_pp = line['from_bus'], line['to_bus']
            from_name = str(net.bus.at[from_pp, 'name'])
            to_name = str(net.bus.at[to_pp, 'name'])

            default = {'A','B','C'}
            p_from = set(sample_power.at[from_pp, 'phase']) if from_pp in sample_power.index else default
            p_to = set(sample_power.at[to_pp, 'phase']) if to_pp in sample_power.index else default
            
            common = sorted(list(p_from.intersection(p_to)))

            if(len(common) == 2): # we dont have 2 phase transformers so we must use line codes with 3 phases and just connect the 2 phases we have.
                common = ['A', 'B', 'C']


            if line_id in sample_imp.index:
                r1 = sample_imp.at[line_id, 'R1_ohm_per_km']
                x1 = sample_imp.at[line_id, 'X1_ohm_per_km']
            else:
                r1, x1 = line['r_ohm_per_km'], line['x_ohm_per_km']
            
            r1, x1 = max(r1, 1e-6), max(x1, 1e-6)
            Zabc_full, _ = carson_modificada_Zabc(r1, x1, DISTANCIAS)
            
            idx = [PHASE_TO_MATRIX_IDX[p] for p in common]
            Z_sub = Zabc_full[np.ix_(idx, idx)]
            
            lcode = f"LCode_{line_id}"
            r_str = format_matrix_for_dss(np.real(Z_sub))
            x_str = format_matrix_for_dss(np.imag(Z_sub))
            
            linecode_cmds.append(f"new linecode.{lcode} nphases={len(common)} rmatrix={r_str} xmatrix={x_str} units=km")
            
        with open(os.path.join(sample_folder, "LineCodes.dss"), 'w') as f:
            f.write("\n".join(linecode_cmds))

        # A3. MV_Lines.dss
        mv_lines_cmds = []
        mv_lines_cmds.append("! MV Feeder Lines")
        for line_id, line in net.line.iterrows():
            # Check if line belongs to ANY MV island
            if line['from_bus'] in mv_nodes_global and line['to_bus'] in mv_nodes_global:
                from_name = net.bus.at[line['from_bus'], 'name']
                to_name = net.bus.at[line['to_bus'], 'name']
                
                p_from = set(sample_power.at[line['from_bus'], 'phase']) if line['from_bus'] in sample_power.index else {'A','B','C'}
                p_to = set(sample_power.at[line['to_bus'], 'phase']) if line['to_bus'] in sample_power.index else {'A','B','C'}
                common = sorted(list(p_from.intersection(p_to)))
                
                if not common: continue
                if len(common) == 2: common = ['A', 'B', 'C'] # 2-phase fallback
                
                nodes = "".join([PHASE_TO_DSS_NODE[p] for p in common])
                
                mv_lines_cmds.append(
                    f"new line.line_{line_id} bus1=bus_{from_name}{nodes} bus2=bus_{to_name}{nodes} "
                    f"linecode=LCode_{line_id} length={0.1*line['length_km']} units=km"
                )

        with open(os.path.join(sample_folder, "MV_Lines.dss"), 'w') as f:
            f.write("\n".join(mv_lines_cmds))


        # ==========================================
        # B. LV Networks (Subfolders)
        # ==========================================
        for t_id, trafo in net.trafo.iterrows():
            if t_id not in lv_islands_map: continue
            
            island_nodes = lv_islands_map[t_id]
            lv_subfolder_name = f"LV_Net_Trafo_{t_id}"
            lv_path = os.path.join(sample_folder, lv_subfolder_name)
            os.makedirs(lv_path)
            
            # --- B1. Transformer.dss ---
            trafo_cmds = []
            hv_bus_id = trafo['hv_bus']
            lv_bus_id = trafo['lv_bus']
            hv_name = net.bus.at[hv_bus_id, 'name']
            lv_name = net.bus.at[lv_bus_id, 'name']
            
            if hv_bus_id in sample_power.index:
                current_phases = set(sample_power.at[hv_bus_id, 'phase'])
            else:
                current_phases = {'A', 'B', 'C'}
            num_phases = len(current_phases)

            kva = trafo['sn_mva'] * 1000.0
            kv_hv = fix_voltage_ll(trafo['vn_hv_kv'])
            kv_lv = fix_voltage_ll(trafo['vn_lv_kv'])
            uk, ur = trafo['vk_percent'], trafo['vkr_percent']
            xhl = np.sqrt(uk**2 - ur**2) if uk >= ur else 0.1

            if num_phases == 1:
                conn_type = "[Wye, Wye]"
                kv_hv_dss = kv_hv / np.sqrt(3)
                kv_lv_dss = kv_lv / np.sqrt(3)
                phase_char = list(current_phases)[0]
                dss_node = PHASE_MAP.get(phase_char, '1')
                hv_bus_str = f"bus_{hv_name}.{dss_node}"
                lv_bus_str = f"bus_{lv_name}.{dss_node}.0"
                phases_param = 1
            else:
                conn_type = "[Delta, Wye]"
                kv_hv_dss = kv_hv
                kv_lv_dss = kv_lv
                hv_bus_str = f"bus_{hv_name}.1.2.3"
                lv_bus_str = f"bus_{lv_name}.1.2.3.0"
                phases_param = 3

            trafo_cmds.append(
                f"new transformer.Trafo_{t_id} phases={phases_param} windings=2 "
                f"buses=[{hv_bus_str}, {lv_bus_str}] Conns={conn_type} "
                f"kvas=[{kva}, {kva}] kvs=[{kv_hv_dss:.4f}, {kv_lv_dss:.4f}] "
                f"Xhl={xhl} %loadloss={ur}"
            )
            with open(os.path.join(lv_path, "Transformer.dss"), 'w') as f:
                f.write("\n".join(trafo_cmds))

            # --- B2. Lines.dss (LV) ---
            lv_lines_cmds = []
            for line_id, line in net.line.iterrows():
                if line['from_bus'] in island_nodes and line['to_bus'] in island_nodes:
                    from_name = net.bus.at[line['from_bus'], 'name']
                    to_name = net.bus.at[line['to_bus'], 'name']
                    
                    p_from = set(sample_power.at[line['from_bus'], 'phase']) if line['from_bus'] in sample_power.index else {'A','B','C'}
                    nodes = "".join([PHASE_TO_DSS_NODE[p] for p in sorted(list(p_from))])
                    
                    lv_lines_cmds.append(
                        f"new line.line_{line_id} bus1=bus_{from_name}{nodes} bus2=bus_{to_name}{nodes} "
                        f"linecode=LCode_{line_id} length={0.1*line['length_km']} units=km"
                    )
            with open(os.path.join(lv_path, "Lines.dss"), 'w') as f:
                f.write("\n".join(lv_lines_cmds))

            # --- B3. Loads.dss (LV) ---
            load_cmds = []
            for bus_id in island_nodes:
                if bus_id not in sample_power.index: continue
                bus_data = net.bus.loc[bus_id]
                # Filter out accidental MV buses in LV island
                if bus_data['vn_kv'] > 0.5: continue 

                b_name = bus_data['name']
                b_power = sample_power.loc[bus_id]
                load_kv = fix_voltage_ll(bus_data['vn_kv'])
                if np.isclose(bus_data['vn_kv'], 0.127): load_kv = 0.127

                for p in ['A', 'B', 'C']:
                    col = f'P_{p}'
                    if p in b_power['phase'] and b_power[col] > 0:
                        kw = b_power[col]
                        load_cmds.append(
                            f"new load.load_{b_name}_{p} bus1=bus_{b_name}{PHASE_TO_DSS_NODE[p]} "
                            f"Phases=1 kv={load_kv:.4f} kw={kw/1000} kvar={kw*P_Q_RATIO/1000} "
                            f"vminpu=0.85 vmaxpu=1.15 model=1"
                        )
            with open(os.path.join(lv_path, "Loads.dss"), 'w') as f:
                f.write("\n".join(load_cmds))

            # --- B4. Master_Local.dss ---
            local_master_cmds = []
            local_master_cmds.append("Clear")
            local_master_cmds.append(f"new circuit.LV_Island_{t_id} basekv={kv_lv_dss:.4f} pu=1.0 phases={phases_param}")
            
            # IMPORTANT: Source must be stiff and grounded same as transformer secondary (wye)
            local_master_cmds.append(
                f"edit Vsource.source bus1={lv_bus_str} basekv={kv_lv_dss:.4f} pu=1.0 phases={phases_param} MVAsc3=1000 MVAsc1=1000"
            )

            # Redirects (Relative Paths)
            local_master_cmds.append("Redirect ../LineCodes.dss") 
            local_master_cmds.append("Redirect Lines.dss")
            local_master_cmds.append("Redirect Loads.dss")
            
            local_master_cmds.append("set controlmode=off")
            local_master_cmds.append(f"Set Voltagebases=[{kv_lv_dss:.4f}]")
            local_master_cmds.append("calcv")
            local_master_cmds.append("Solve")

            with open(os.path.join(lv_path, "Master_Local.dss"), 'w') as f:
                f.write("\n".join(local_master_cmds))

            # Add to Global Master
            master_redirects.append(f"Redirect {lv_subfolder_name}/Transformer.dss")
            master_redirects.append(f"Redirect {lv_subfolder_name}/Lines.dss")
            master_redirects.append(f"Redirect {lv_subfolder_name}/Loads.dss")

        # ==========================================
        # C. Master.dss (Global)
        # ==========================================
        master_cmds = []
        master_cmds.append("Clear")
        
        # Order matters! VSource -> LineCodes -> MV Lines -> LV Clusters
        master_redirects.insert(0, "Redirect MV_Lines.dss")
        master_redirects.insert(0, "Redirect LineCodes.dss")
        master_redirects.insert(0, "Redirect VSource.dss")
        
        for cmd in master_redirects:
            master_cmds.append(f"{cmd}")
        
        master_cmds.append("set controlmode=off")
        master_cmds.append(f"Set Voltagebases={voltage_bases_str}")
        master_cmds.append("calcv")
        master_cmds.append("Solve")

        with open(os.path.join(sample_folder, "Master.dss"), 'w') as f:
            f.write("\n".join(master_cmds))
            
    print("\nGeneration Complete.")