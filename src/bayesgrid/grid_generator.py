# Standard Library
import os
import pickle
import warnings
from collections import deque

# Data Science & Numerics
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
from scipy.stats import weibull_min

# PyMC & PyTensor
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.pytensorf import collect_default_updates
from pytensor.compile.mode import get_mode

# Power Systems & GIS
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
import osmnx as ox

# Plotting
import matplotlib.pyplot as plt


# fetch data
from ._registry import fetch_trace  





###################### AUXILIARY FUNCTIONS (PICKLE) ######################

# Open a file in write mode
def save_pickle(file_object,file_name):
    with open(file_name, 'wb') as file:
        # Serialize and save the object to the file
        pickle.dump(file_object, file)

def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        # Deserialize and load the object from the file
        loaded_object = pickle.load(file)

    return loaded_object



def find_ramification_orders_v2(G, source):
    """
    Enhanced version:
    - For each ramification node r_node (degree > 2),
    - Explore all its neighbors.
    - If a neighbor is not on the path from r_node to the source,
      treat it as a new ramification node.
    - Its parent is the first ramification on the path to the source (if any), or -1.
    
    Returns:
        ramification_orders: dict {node: order}
        ramification_parents: dict {node: parent_node}
    """
    ramifications = {node for node in G.nodes if G.degree(node) > 2}
    ramification_orders = {}
    ramification_parents = {}

    for r_node in ramifications:
        try:
            # Shortest path from r_node to source
            path = nx.shortest_path(G, source=source, target=r_node)
            path_set = set(path)
            ramification_path = [node for node in path[:-1] if node in ramification_orders.keys()]

            order = len(ramification_path)
            parent = ramification_path[-1] if ramification_path else -1

            # Register the main ramification node
            if(len(ramification_orders.keys())==0):
                ramification_orders[r_node] = order
                ramification_parents[r_node] = parent

            # Now check each neighbor to see if it's a new branch
            for neighbor in G.neighbors(r_node):
                if neighbor not in path_set:
                    # This is a new ramification node
                    if neighbor not in ramification_orders:
                        path = nx.shortest_path(G, source=source, target=neighbor)
                        path_set = set(path)
                        ramification_path = [node for node in path[:-1] if node in ramification_orders.keys()]
                        order = len(ramification_path)
                        parent = ramification_path[-1] if ramification_path else -1
                        ramification_orders[neighbor] = order + 1
                        ramification_parents[neighbor] = parent

        except nx.NetworkXNoPath:
            continue

    return ramification_orders, ramification_parents




def find_longest_path_from_source(G, source):
    """
    Brute-force method to find the longest simple path from source
    """
    all_paths = []
    for target in G.nodes:
        try:
            paths = list(nx.all_simple_paths(G, source=source, target=target))
            all_paths.extend(paths)
        except nx.NetworkXNoPath:
            continue
    if not all_paths:
        return []
    return max(all_paths, key=len)


def find_ramification_orders_v3(G, source):
    """
    R3 version:
    - Main path = longest simple path starting from source
    - No node on main path can be a ramification
    - Neighbors branching from main path are considered ramifications
    - Parent = nearest previous ramification (or main path node)
    """
    main_path = find_longest_path_from_source(G, source)
    main_path_set = set(main_path)

    ramification_orders = {}
    ramification_parents = {}

    visited = set(main_path)  # already part of main path
    order_counter = 1  # start orders from 1 for branches

    for idx, node in enumerate(main_path):
        for neighbor in G.neighbors(node):
            if neighbor not in main_path_set and neighbor not in visited:
                # This is a new ramification
                ramification_orders[neighbor] = order_counter
                ramification_parents[neighbor] = node
                visited.add(neighbor)

                # Optional: Traverse this branch and add further ramifications
                queue = [(neighbor, order_counter, node)]
                while queue:
                    current, current_order, parent = queue.pop(0)
                    for nbr in G.neighbors(current):
                        if nbr not in visited:
                            ramification_orders[nbr] = current_order + 1
                            ramification_parents[nbr] = current
                            visited.add(nbr)
                            queue.append((nbr, current_order + 1, current))

    return ramification_orders, ramification_parents, main_path



def find_main_path_and_ramifications(G, source):
    # Step 1: Find shortest paths from source to all reachable nodes
    lengths = nx.single_source_shortest_path_length(G, source)
    furthest_node = max(lengths, key=lengths.get)

    # Step 2: Get the shortest path to the furthest node => main path
    main_path = nx.shortest_path(G, source=source, target=furthest_node)
    main_path_set = set(main_path)

    # Step 3: Choose method based on graph type
    get_neighbors = G.successors if G.is_directed() else G.neighbors

    # Step 4: Identify ramifications
    ramifications = []
    for node in main_path:
        for neighbor in get_neighbors(node):
            if neighbor not in main_path_set:
                ramifications.append((node, neighbor))

    return main_path, ramifications


def first_ramification_in_path(path, ramification_keys):
    ramification_set = set(ramification_keys)  # O(1) lookups
    for idx, node in enumerate(path):
        if node in ramification_set:
            return node
    return -1  # or raise an error if no ramification found




################################################### BAYESIAN MODELS - POWER AND PHASE 

# ====================================================================
# --- Part 1: Data Preprocessing Utility (Unchanged) ---
# ====================================================================

def preprocess_power_and_phase_data(df, n_discrete=10, 
                    hop_col='hop_distance_normalized', 
                    phase_col='phases', 
                    power_cols=['P_A', 'P_B', 'P_C']):
    """
    Preprocesses a raw DataFrame for the BayesianPowerModel.
    ... (code from previous step, unchanged) ...
    """
    print("Preprocessing data...")
    if hop_col not in df.columns:
        raise ValueError(f"Missing required column: {hop_col}")
    if phase_col not in df.columns:
        raise ValueError(f"Missing required column: {phase_col}")

    df_proc = df.copy()

    # --- 1. Discretize Hop Distance ---
    df_proc['hop_distance_discrete'] = pd.cut(
        df_proc[hop_col], 
        bins=n_discrete, 
        labels=False, # Use integer labels 0 to n_discrete-1
        include_lowest=True,
        duplicates='drop'
    )
    
    if df_proc['hop_distance_discrete'].isnull().any():
        warnings.warn("NaNs found in 'hop_distance_discrete' after binning. Filling with zone 0.")
        df_proc['hop_distance_discrete'] = df_proc['hop_distance_discrete'].fillna(0)

    df_proc['hop_zone_idx'] = pd.Categorical(df_proc['hop_distance_discrete']).codes
    hop_zone_idx = df_proc['hop_zone_idx'].values.astype(int)
    n_zones = len(np.unique(hop_zone_idx))

    # --- 2. Encode Phases ---
    phase_map = {'A': 0, 'B': 1, 'C': 2, 'AB': 3, 'CA': 4, 'BC': 5, 'ABC': 6}
    

    phase_category_idx = df_proc[phase_col].map(phase_map).fillna(-1).astype(int)
    
    # --- 3. Filter out invalid data ---
    valid_mask = (phase_category_idx != -1)
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        warnings.warn(f"Dropping {n_dropped} buses with unknown phase categories.")
        df_proc = df_proc[valid_mask]
        hop_zone_idx = hop_zone_idx[valid_mask]
        phase_category_idx = phase_category_idx[valid_mask]

    # --- 4. Get Power Array ---
    if all(col in df_proc.columns for col in power_cols):
         power_array = df_proc[power_cols].values
         if np.isnan(power_array).any():
             n_nans = np.isnan(power_array).sum()
             warnings.warn(f"Found {n_nans} NaNs in power data. Replacing with 0.0.")
             power_array = np.nan_to_num(power_array, nan=0.0)
    else:
        warnings.warn(f"Power columns {power_cols} not found. Returning None for power array.")
        power_array = None

    print(f"Preprocessing complete. {len(hop_zone_idx)} valid buses found across {n_zones} zones.")
    return hop_zone_idx, phase_category_idx, power_array, n_zones


# ====================================================================
# --- Part 2: The Bayesian Power Model Class (UPDATED) ---
# ====================================================================

class BayesianPowerModel:
    """
    A class to learn and generate synthetic power distribution system
    parameters (phase allocation and power) based on hop distance and
    network graph topology.
    """
    
    def __init__(self, trace_path=None, total_demand=None):
        """
        Initializes the Bayesian Power Model.

        Args:
            trace_path (str, optional): Path to a pre-computed trace file
                                        (e.g., 'trace.pickle').
        """
        self.model = None
        self.trace = None
        self.n_train_buses = None
        self.n_zones = None
        self.phase_categories = ['A', 'B', 'C', 'AB', 'BC', 'CA', 'ABC']
        self.phase_coords = ["A", "B", "C"]

        if(trace_path==None):
            print("No trace_path provided. Fetching default pre-trained model...")
            trace_path = fetch_trace("power")

        self.total_demand = None

        if trace_path:
            print(f"Loading trace from {trace_path}...")
            try:
                self.trace = az.from_netcdf(trace_path)
                
                self.n_train_buses = self.trace['observed_data'].bus.shape[0]
                self.n_zones = len(list(set(self.trace['constant_data'].hop_zone_idx.values)))
                print(f"Successfully loaded pre-trained model.")
                print(f"Model was trained with {self.n_train_buses} buses and {self.n_zones} zones.")
            except Exception as e:
                warnings.warn(f"Could not load trace from {trace_path}. Error: {e}")
                print("You must call .learn() with data to train a new model.")
        else:
            print("No trace path provided. Call .learn() to train a new model.")
            
        # --- Define the static transition tensor ---
        # Original allowed transitions (ragged)
        allowed_transitions_dict = {
                        0: [0], # A->A
                        1: [1], # B->B
                        2: [2], # C-> C
                        3: [0,1,3], #AB -> (A,B,AB)
                        4: [1,2,4], #BC -> (B,C,BC) 
                        5: [2,0,5], #CA -> (C,A,CA)
                        6: [0,1,2,3,4,5,6]
                    }
        # Pad to rectangular shape using -1 as a sentinel
        max_len = max(len(v) for v in allowed_transitions_dict.values())
        padded_transitions = np.full((7, max_len), -1, dtype="int64")
        for k, v in allowed_transitions_dict.items():
            padded_transitions[k, :len(v)] = v
        
        # Create tensor from padded array
        self.transitions_tensor = pt.as_tensor(padded_transitions)


    def _build_model(self, n_buses, n_zones, hop_zone_idx_data, phase_data, power_data):
        """
        Internal method to define the PyMC model structure.
        ... (code from previous step, unchanged) ...
        """
        coords = {
            "bus": range(n_buses),
            "phase_category": self.phase_categories,
            "phase": self.phase_coords,
            "hop_zone": range(n_zones)
        }

        with pm.Model(coords=coords) as model:
            # --- Data Placeholder ---
            hop_zone_idx = pm.MutableData("hop_zone_idx", hop_zone_idx_data, dims="bus")

            # --- Part 1: Phase Allocation ---
            a_by_zone = pm.HalfNormal("a_by_zone", sigma=1, dims=("hop_zone", "phase_category"))
            a_for_each_bus = a_by_zone[hop_zone_idx]
            probs = pm.Dirichlet("probs", a=a_for_each_bus, dims=("bus", "phase_category"))

            phase_choice = pm.Categorical(
                "phase_likelihood", p=probs, observed=phase_data, dims="bus"
            )

            # --- Part 2: Hierarchical Power Generation ---
            alpha_hp, beta_hp = 1.0, 1.0
            
            alpha_mono = pm.Gamma("alpha_mono", alpha=alpha_hp, beta=beta_hp)
            beta_mono = pm.Gamma("beta_mono", alpha=alpha_hp, beta=beta_hp)
            P_potential_mono = pm.Gamma("P_potential_mono", alpha=alpha_mono, beta=beta_mono)

            alpha_bi = pm.Gamma("alpha_bi", alpha=alpha_hp, beta=beta_hp)
            beta_bi = pm.Gamma("beta_bi", alpha=alpha_hp, beta=beta_hp)
            P_potential_bi_total = pm.Gamma("P_potential_bi_total", alpha=alpha_bi, beta=beta_bi)
            split_factor_bi = pm.Beta("split_factor_bi", alpha=2.0, beta=2.0)
            
            alpha_tri = pm.Gamma("alpha_tri", alpha=alpha_hp, beta=beta_hp)
            beta_tri = pm.Gamma("beta_tri", alpha=alpha_hp, beta=beta_hp)
            P_potential_tri_total = pm.Gamma("P_potential_tri_total", alpha=alpha_tri, beta=beta_hp)
            split_factor_tri = pm.Dirichlet("split_factor_tri", a=np.array([2.0, 2.0, 2.0]), dims="phase")

            # --- Part 3: Linking and Likelihood ---
            zeros_bus = pt.zeros(n_buses)
            p_mono_vec = pt.full(n_buses, P_potential_mono)
            p_bi_vec = pt.full(n_buses, P_potential_bi_total)
            
            p_cat_A = pt.stack([p_mono_vec, zeros_bus, zeros_bus], axis=1)
            p_cat_B = pt.stack([zeros_bus, p_mono_vec, zeros_bus], axis=1)
            p_cat_C = pt.stack([zeros_bus, zeros_bus, p_mono_vec], axis=1)
            p_cat_AB = pt.stack([p_bi_vec * split_factor_bi, p_bi_vec * (1 - split_factor_bi), zeros_bus], axis=1)
            p_cat_AC = pt.stack([p_bi_vec * split_factor_bi, zeros_bus, p_bi_vec * (1 - split_factor_bi)], axis=1)
            p_cat_BC = pt.stack([zeros_bus, p_bi_vec * split_factor_bi, p_bi_vec * (1 - split_factor_bi)], axis=1)
            
            p_tri_vec = pt.full(n_buses, P_potential_tri_total)
            p_cat_ABC = p_tri_vec[:, None] * split_factor_tri[None, :]

            potential_power_by_category = pt.stack([p_cat_A, p_cat_B, p_cat_C, p_cat_AB, p_cat_AC, p_cat_BC, p_cat_ABC], axis=1)
            
            mu_p = pm.Deterministic(
                "mu_p",
                potential_power_by_category[pt.arange(n_buses), phase_choice],
                dims=("bus", "phase")
            )

            sigma_p = pm.HalfNormal("sigma_p", sigma=1.0)
            
            power_observed_likelihood = pm.TruncatedNormal(
                "power_observed",
                mu=mu_p,
                sigma=sigma_p,
                observed=power_data, # Can be None for prediction
                dims=("bus", "phase"),
                lower=0.0 # Assuming power cannot be negative
            )
        return model

    def learn(self, hop_zone_idx, observed_phases, observed_power_p, n_zones=None, **sample_kwargs):
        """
        Trains the Bayesian model on new data.
        ... (code from previous step, unchanged) ...
        """
        self.n_train_buses = len(hop_zone_idx)
        if n_zones is None:
            self.n_zones = int(np.max(hop_zone_idx)) + 1
        else:
            self.n_zones = n_zones
        
        print(f"Building model for learning with {self.n_train_buses} buses and {self.n_zones} zones...")

        self.model = self._build_model(
            n_buses=self.n_train_buses,
            n_zones=self.n_zones,
            hop_zone_idx_data=hop_zone_idx,
            phase_data=observed_phases,
            power_data=observed_power_p
        )
        
        sample_args = {'draws': 1000, 'tune': 1000, 'cores': 4}
        sample_args.update(sample_kwargs)
        
        print(f"Starting sampling with args: {sample_args}")
        with self.model:
            self.trace = pm.sample(**sample_args)
        
        print("Learning complete. Trace is stored in self.trace.")
        return self.trace

    def _generate_unconstrained_data(self, new_hop_zone_idx, random_seed=None):
        """
        (Internal) Generates unconstrained synthetic phase and power data.
        """
        if self.trace is None:
            raise ValueError("No trace found. Please load a trace or run .learn() first.")
        
        new_hop_zone_idx = np.asarray(new_hop_zone_idx)
        n_new_buses = len(new_hop_zone_idx)
        
        if np.max(new_hop_zone_idx) >= self.n_zones:
            warnings.warn(f"new_hop_zone_idx contains values >= {self.n_zones} (max trained zone). "
                          "Results may be unpredictable.")

        print(f"Step 1: Generating unconstrained data for {n_new_buses} buses...")

        # Build a model instance with the *original* training dimensions
        self.model = self._build_model(
            n_buses=self.n_train_buses,
            n_zones=self.n_zones,
            hop_zone_idx_data=np.zeros(self.n_train_buses, dtype=int), 
            phase_data=None,
            power_data=None
        )
        
        posterior_predictive_new_network = None
        
        with self.model:
            if n_new_buses <= self.n_train_buses:
                # --- Case 1: New network is smaller or equal ---
                padding_size = self.n_train_buses - n_new_buses
                padding_values = np.random.randint(0, self.n_zones, size=padding_size)
                padded_hop_zone_idx = np.concatenate([new_hop_zone_idx, padding_values])
                
                pm.set_data({"hop_zone_idx": padded_hop_zone_idx})
                
                posterior_predictive_full = pm.sample_posterior_predictive(
                    self.trace,
                    var_names=["phase_likelihood", "power_observed"],
                    random_seed=random_seed,
                )
                
                posterior_predictive_new_network = posterior_predictive_full.posterior_predictive.isel(
                    bus=slice(0, n_new_buses)
                )

            else:
                # --- Case 2: New network is larger ---
                ppc_chunks = []
                for i in range(0, n_new_buses, self.n_train_buses):
                    chunk = new_hop_zone_idx[i : i + self.n_train_buses]
                    current_chunk_size = len(chunk)

                    if current_chunk_size < self.n_train_buses:
                        padding_size = self.n_train_buses - current_chunk_size
                        padding_values = np.random.randint(0, self.n_zones, size=padding_size)
                        chunk = np.concatenate([chunk, padding_values])
                    
                    pm.set_data({"hop_zone_idx": chunk})
                    
                    ppc_chunk_full = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=["phase_likelihood", "power_observed"],
                        random_seed=random_seed
                    )
                    
                    ppc_chunks.append(
                        ppc_chunk_full.posterior_predictive.isel(bus=slice(0, current_chunk_size))
                    )
                
                posterior_predictive_new_network = xr.concat(ppc_chunks, dim="bus")

        print("Step 1 complete.")
        
        # --- Format and return the output ---
        gen_phases_xr = posterior_predictive_new_network["phase_likelihood"]
        gen_power_xr = posterior_predictive_new_network["power_observed"]

        # Combine chains and draws into a single 'sample' dimension
        generated_phases = gen_phases_xr.stack(sample=("chain", "draw")).transpose("sample", "bus").values
        generated_power = gen_power_xr.stack(sample=("chain", "draw")).transpose("sample", "bus", "phase").values

        return generated_phases, generated_power, gen_phases_xr
    
    @staticmethod
    def _find_ramification_nodes_bfs(graph, source):
        """
        Finds all ramification nodes, their BFS order, and their ramification parent.
        A ramification node is the source, a leaf (degree 1), or a branching
        point (degree > 2).
        """
        print("Step 2a: Analyzing graph topology...")
        
        full_parents = {source: None}
        q = deque([source])
        visited = {source}
        
        ramification_orders = {source: 0}
        ramification_parents = {source: None}
        
        # Find all leaf nodes (excluding source if it's a leaf)
        leaves = {n for n in graph.nodes if graph.degree(n) == 1 and n != source}
        
        order_counter = 1
        
        while q:
            curr = q.popleft()
            
            for neighbor in graph.neighbors(curr):
                if neighbor not in visited:
                    visited.add(neighbor)
                    full_parents[neighbor] = curr
                    q.append(neighbor)
                    
                    # Check if this neighbor is a ramification node
                    is_ramification = (
                        graph.degree(neighbor) != 2 or 
                        neighbor in leaves
                    )
                    
                    if is_ramification:
                        ramification_orders[neighbor] = order_counter
                        order_counter += 1
                        
                        # Find its parent in the ramification set by tracing back
                        parent_ram = curr
                        while parent_ram is not None and parent_ram not in ramification_orders:
                            parent_ram = full_parents[parent_ram]
                        ramification_parents[neighbor] = parent_ram
                        
        print(f"Found {len(ramification_orders)} ramification nodes.")
        return ramification_orders, ramification_parents

    @staticmethod
    def _find_first_ramification_in_path(path, ramification_keys):
        """
        Helper to find the first node in a path that is a ramification node.
        """
        # Start from path[1] to skip the node itself
        for node in path[1:]:
            if node in ramification_keys:
                return node
        # This case should ideally not be hit if source is a ramification node
        return path[-1] 

    def generate_data(self, net,new_hop_zone_idx=None,
                                    random_seed=None, scan_draws=1000, scan_tune=0):
            """
            Generates graph-consistent synthetic phase and power data.

            This is the primary method for users.

            Args:
                new_hop_zone_idx (list or np.array): A list/array of hop zone indices
                                                    for the new synthetic grid.
                                                    Must be in node-order (len == n_nodes).
                graph (nx.Graph): The networkx graph of the new grid.
                source_bus_idx (int): The node index of the source/slack bus.
                random_seed (int, optional): Seed for reproducible generation.
                scan_draws (int, optional): Number of draws for the scan model.
                scan_tune (int, optional): Number of tune steps for the scan model.

            Returns:
                tuple: A tuple containing:
                    - consistent_phase_samples (np.array): 
                        Shape (n_scan_samples, n_buses).
                        The new, graph-consistent phase category indices (0-6).
                    - unconstrained_power_samples (np.array): 
                        Shape (n_bhm_samples, n_buses, 3).
                        The *original* power samples.
            """
            # --- Validate Inputs ---
            # --- 1. Calculate Hop Distance for all buses ---

            # Keep these for the next steps
            source_bus_idx = net.ext_grid.bus.iloc[0] # This will be 0
            graph = pn.create_nxgraph(net)



            substation_node = net.bus.name.iloc[source_bus_idx] # Get original osmnx node ID
            if(new_hop_zone_idx==None):
                hop_distances_new_net = {}
                for bus_idx in graph.nodes:
                    try:
                        dist = nx.shortest_path_length(graph, source=source_bus_idx, target=bus_idx)
                        hop_distances_new_net[bus_idx] = dist
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        hop_distances_new_net[bus_idx] = np.nan

                hop_series = pd.Series(hop_distances_new_net, index=range(len(graph.nodes)))
                max_dist = hop_series.max()
                hop_series = hop_series.fillna(max_dist)

                # --- 3. Discretize Bus Hop Distances for POWER (10 Bins) ---
                N_BINS_POWER_IMPD = 10
                hop_zone_series_power = pd.cut(
                    hop_series, bins=N_BINS_POWER_IMPD, labels=False, include_lowest=True
                )
                new_hop_zone_idx = hop_zone_series_power.values
                print(f"Created {len(new_hop_zone_idx)} bus zone indices for power (10 bins).")


            if len(new_hop_zone_idx) != len(graph.nodes):
                raise ValueError(
                    f"Length of 'new_hop_zone_idx' ({len(new_hop_zone_idx)}) does not "
                    f"match the number of nodes in 'graph' ({len(graph.nodes)})."
                )

            # ====================================================================
            # --- Step 1: Run original BHM to get unconstrained probabilities ---
            # ====================================================================
            gen_phases, gen_power, gen_phases_xr = self._generate_unconstrained_data(
                new_hop_zone_idx, random_seed
            )
            
            # Calculate the probabilities for each phase at each bus
            print("Step 2: Calculating phase probabilities...")
            n_chains, n_draws, n_buses = gen_phases_xr.shape
            total_samples = n_chains * n_draws
            
            counts = gen_phases_xr.to_dataframe().groupby('bus')['phase_likelihood'].value_counts().unstack(fill_value=0)
            counts = counts.reindex(columns=range(7), fill_value=0)
            counts = counts.reindex(range(n_buses), fill_value=0)
            
            probs_phase_likelihood = (counts / total_samples).values
            probs_phase_likelihood = np.maximum(probs_phase_likelihood, 1e-9)
            probs_phase_likelihood = probs_phase_likelihood / probs_phase_likelihood.sum(axis=1, keepdims=True)

            # ====================================================================
            # --- Step 2: Prepare Graph Tensors for Scan Model ---
            # ====================================================================
            ram_orders, ram_parents = self._find_ramification_nodes_bfs(graph, source_bus_idx)
            
            sorted_ram_dict = dict(sorted(ram_orders.items(), key=lambda item: item[1]))
            
            list_ram_nodes = list(sorted_ram_dict.keys())
            N_ram = len(list_ram_nodes)
            
            probs_ram_nodes = probs_phase_likelihood[list_ram_nodes, :]
            phase_probabilities_tensor = pt.as_tensor_variable(probs_ram_nodes)

            sorted_parents_idx_list = []
            for node_idx in list_ram_nodes[1:]:
                parent_node = ram_parents[node_idx]
                if parent_node in list_ram_nodes:
                    sorted_parents_idx_list.append(list_ram_nodes.index(parent_node))
                else:
                    sorted_parents_idx_list.append(-1) 
                    
            sorted_parents_idx_array = pt.as_tensor(
                np.array(sorted_parents_idx_list, dtype="int64")
            )

            # ====================================================================
            # --- Step 3: Define and Run the Scan Model ---
            # ====================================================================
            N = N_ram
            print(f"Step 3: Building graph-consistency scan model for {N} ramification nodes...")
            
            coords={'samples':range(N_ram)}

            y=np.random.randint(0,2,10)

            with pm.Model(coords=coords) as scan_model:
                
                sampled_tensor_init = pt.full((N,),-1)
                
                alpha=pm.HalfNormal('alpha',sigma=5)
                beta=pm.HalfNormal('beta',sigma=5)
                #y_init=pm.math.constant(6) #starting with 3 phase
                y_group_init = pt.constant(6, dtype="int64")
                y_init = pt.constant(1, dtype="int64")
                count_value_init = pt.constant(0,dtype='int64')
                
                #data=pm.MutableData('data',y)

                def step(*args):
                    #order of args is sequence,outputs_info,non_sequences
                    y_prev, y_group, count_value, sampled_tensor, alpha, beta, = args

                    sampled_tensor=pt.set_subtensor(sampled_tensor[count_value],y_group)

                    #probs=pm.Dirichlet('probs',a=np.ones(k),dims=('parent_phase','phase'))

                    #pm.math.eq(y_prev,)

                    #final_phase=pm.Categorical('final_phase',probs[parent_phase_idx],dims='buses')
                    

                    # prob_3_phi=pm.math.constant(1)  
                    # prob_2_phi=pm.math.constant(1)
                    # prob_1_phi=pm.math.constant(1)

                    # (NEw)
                    prob_0 = phase_probabilities_tensor[count_value][0]
                    prob_1 = phase_probabilities_tensor[count_value][1]
                    prob_2 = phase_probabilities_tensor[count_value][2]
                    prob_3 = phase_probabilities_tensor[count_value][3]
                    prob_4 = phase_probabilities_tensor[count_value][4]
                    prob_5 = phase_probabilities_tensor[count_value][5]
                    prob_6 = phase_probabilities_tensor[count_value][6]

                    def compute_real_prob(k, y_last_order, prob_sampled):
                        # Fetch allowed values for current y_prev
                        allowed_k = self.transitions_tensor[y_last_order]

                        # Check if current k is in allowed_k
                        is_allowed = pt.any(pt.eq(allowed_k, k))

                        # Return prob_sampled if allowed, else 0
                        return pt.switch(is_allowed, prob_sampled, 0.0)


                    #def compute_real_prob(k):
                    #    allowed_k = pt.as_tensor(np.array(allowed_transitions[k], dtype="int64"))
                    #    is_allowed = pt.any(pt.eq(allowed_k, y_group))
                    #    return pt.switch(is_allowed, prob_sampled, 0.0)


                    parent=sorted_parents_idx_array[count_value]
                    y_last_order=pt.switch(pt.eq(parent,-1),0,sampled_tensor[parent])

                    #real_prob_0=pm.math.switch(pm.math.eq(y_prev,0),1,prob_sampled)
                    real_prob_0=compute_real_prob(0,y_last_order,prob_0)
                    real_prob_1=compute_real_prob(1,y_last_order,prob_1)
                    real_prob_2=compute_real_prob(2,y_last_order,prob_2)
                    real_prob_3=compute_real_prob(3,y_last_order,prob_3)
                    real_prob_4=compute_real_prob(4,y_last_order,prob_4)
                    real_prob_5=compute_real_prob(5,y_last_order,prob_5)
                    real_prob_6=compute_real_prob(6,y_last_order,prob_6)

                    

                    sum_prob = real_prob_0 + real_prob_1 + real_prob_2 + real_prob_3 + real_prob_4 + real_prob_5 + real_prob_6

                    norm_prob_0 = real_prob_0 / sum_prob
                    norm_prob_1 = real_prob_1 / sum_prob
                    norm_prob_2 = real_prob_2 / sum_prob
                    norm_prob_3 = real_prob_3 / sum_prob
                    norm_prob_4 = real_prob_4 / sum_prob
                    norm_prob_5 = real_prob_5 / sum_prob
                    norm_prob_6 = real_prob_6 / sum_prob


                    #p_next = at.stack([norm_prob_0, norm_prob_1, norm_prob_2, norm_prob_3])
                    

                    y=pm.Bernoulli.dist(p=real_prob_0)


                    y_group=pm.Categorical.dist(p=[norm_prob_0, norm_prob_1, norm_prob_2, norm_prob_3,norm_prob_4,norm_prob_5,norm_prob_6])

                    
                    
                    count_value+=1

                    

                    return (y,y_group,count_value,sampled_tensor), collect_default_updates([y_prev,alpha,beta,y,y_group])


                [y_hat,y_group,count_value,sampled_tensor],updates = pytensor.scan(
                    fn=step,
                    n_steps=N-1,
                    outputs_info=[y_init,y_group_init,count_value_init,sampled_tensor_init],
                    non_sequences=[alpha, beta],
                    strict=True,
                    mode = get_mode(None) # use get_mode("JAX") if you try to sample with numpyro
                )

                #pm.Deterministic('y',y_hat)
                pm.Deterministic('y_group',y_group)
                #pm.Deterministic('count_value',count_value)

                #pm.Deterministic('sampled_tensor',sampled_tensor[-1])
            # ====================================================================
            # --- Step 4: Map Ramification Phases to Full Grid ---
            # ====================================================================
            with scan_model:
                trace_scan = pm.sample(draws=scan_draws,tune=0,cores=4)
            print("Step 5: Mapping consistent phases to full grid...")
            # (This part is unchanged)

            
            y_group_samples_flat = trace_scan.posterior['y_group'].stack(
                sample=("chain", "draw")
            ).transpose("sample", "y_group_dim_0").values

            n_scan_samples = y_group_samples_flat.shape[0]
            
            source_phase_col = np.full((n_scan_samples, 1), 6, dtype=int)
            phase_samples_ram = np.hstack([source_phase_col, y_group_samples_flat])
            
            ram_keys = set(ram_orders.keys())
            
            all_nodes_phases_array = np.zeros((n_scan_samples, n_buses), dtype=int)

            shortest_paths = {}
            for node_idx in graph.nodes:
                if node_idx != source_bus_idx and node_idx not in ram_keys:
                    shortest_paths[node_idx] = nx.shortest_path(graph, node_idx, source_bus_idx)
            
            for i in range(n_scan_samples):
                color_dict = dict(zip(list_ram_nodes, phase_samples_ram[i, :]))
                
                sample_phases = {}
                for node_idx in graph.nodes:
                    if node_idx == source_bus_idx:
                        sample_phases[node_idx] = 6 # Source is ABC
                    elif node_idx in ram_keys:
                        sample_phases[node_idx] = color_dict[node_idx]
                    else:
                        path = shortest_paths[node_idx]
                        first_ram_node = self._find_first_ramification_in_path(path, ram_keys)
                        sample_phases[node_idx] = color_dict[first_ram_node]
                
                for node_idx in range(n_buses):
                    all_nodes_phases_array[i, node_idx] = sample_phases[node_idx]

            print("--- Generation Complete ---")


            # ====================================================================
            # --- Step 5: Making the power consistent ---
            # ====================================================================
            print("Step 6: Re-allocating power based on consistent phases...")
            
            n_scan_samples, n_buses = all_nodes_phases_array.shape
            n_bhm_samples = gen_power.shape[0]
            
            if n_scan_samples > n_bhm_samples:
                warnings.warn(f"Generating {n_scan_samples} consistent phase samples, "
                            f"but only {n_bhm_samples} unconstrained power samples are available. "
                            "Power samples will be reused.")
            
            # Create the new power array
            consistent_power_samples = np.zeros((n_scan_samples, n_buses, 3))
            
            # Loop over each new *consistent* sample
            for i in range(n_scan_samples):
                # Pick a corresponding unconstrained power sample (reuse if needed)
                power_sample_idx = i % n_bhm_samples
                original_power_sample = gen_power[power_sample_idx, :, :]
                
                # Determine the target total demand for this sample
                original_total_demand = original_power_sample.sum()
                target_total_demand = original_total_demand
                if self.total_demand is not None:
                    target_total_demand = self.total_demand
                
                # Loop over each bus
                for bus in range(n_buses):
                    original_power_bus = original_power_sample[bus, :] # (pA, pB, pC)
                    original_bus_demand = original_power_bus.sum()
                    
                    # Calculate this bus's ratio of the *original* total demand
                    bus_demand_ratio = original_bus_demand / (original_total_demand + 1e-9)
                    
                    # This is the new total demand for *this bus*
                    new_bus_total_demand = bus_demand_ratio * target_total_demand
                    
                    # Get the new consistent phase for this bus
                    phase = all_nodes_phases_array[i, bus]
                    
                    if phase == 0: # A
                        consistent_power_samples[i, bus, 0] = new_bus_total_demand
                    
                    elif phase == 1: # B
                        consistent_power_samples[i, bus, 1] = new_bus_total_demand
                    
                    elif phase == 2: # C
                        consistent_power_samples[i, bus, 2] = new_bus_total_demand

                    elif phase == 3: # AB
                        denom = original_power_bus[0] + original_power_bus[1] + 1e-9
                        ratio_a = original_power_bus[0] / denom
                        ratio_b = original_power_bus[1] / denom
                        consistent_power_samples[i, bus, 0] = ratio_a * new_bus_total_demand
                        consistent_power_samples[i, bus, 1] = ratio_b * new_bus_total_demand

                    elif phase == 4: # BC
                        denom = original_power_bus[1] + original_power_bus[2] + 1e-9
                        ratio_b = original_power_bus[1] / denom
                        ratio_c = original_power_bus[2] / denom
                        consistent_power_samples[i, bus, 1] = ratio_b * new_bus_total_demand
                        consistent_power_samples[i, bus, 2] = ratio_c * new_bus_total_demand
                    
                    elif phase == 5: # CA

                        denom = original_power_bus[0] + original_power_bus[2] + 1e-9
                        ratio_a = original_power_bus[0] / denom
                        ratio_c = original_power_bus[2] / denom
                        consistent_power_samples[i, bus, 0] = ratio_a * new_bus_total_demand
                        consistent_power_samples[i, bus, 2] = ratio_c * new_bus_total_demand

                    elif phase == 6: # ABC
                        # Re-distribute the new total bus demand based on the
                        # original 3-phase ratios
                        denom = original_bus_demand + 1e-9
                        ratio_a = original_power_bus[0] / denom
                        ratio_b = original_power_bus[1] / denom
                        ratio_c = original_power_bus[2] / denom
                        consistent_power_samples[i, bus, 0] = ratio_a * new_bus_total_demand
                        consistent_power_samples[i, bus, 1] = ratio_b * new_bus_total_demand
                        consistent_power_samples[i, bus, 2] = ratio_c * new_bus_total_demand

            print("--- Generation Complete ---")
            
            # --- Return the new phases and the NEW consistent power ---
            return all_nodes_phases_array, consistent_power_samples

    def save_trace(self, file_path):
        """
        Saves the model's trace to a NetCDF file.

        Args:
            file_path (str): The destination file path (e.g., 'new_trace.nc').
        """
        if self.trace:
            self.trace.to_netcdf(file_path)
            print(f"Trace saved to {file_path}")
        else:
            print("No trace to save. Run .learn() first.")
            

    def get_phase_map(self):
        """Returns the mapping of phase category indices to names."""
        return {i: name for i, name in enumerate(self.phase_categories)}





# ====================================================================
# --- Part 1: Data Preprocessing Utility ---
# ====================================================================

def preprocess_duration_data(df, n_discrete=10, 
                             hop_col='hop_distance_normalized', 
                             duration_col='duration'):
    """
    Preprocesses a raw DataFrame for the BayesianDurationModel.

    This function:
    1. Discretizes a 'hop_distance' column into zones.
    2. Creates the 'is_positive' (hurdle) array.
    3. Filters data for positive durations and their corresponding zones.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_discrete (int): Number of bins for hop distance.
        hop_col (str): Column name for normalized hop distance.
        duration_col (str): Column name for the duration data.

    Returns:
        tuple: A tuple containing:
            - hop_zone_idx_all (np.array): Zone index for *all* buses.
            - is_positive_all (np.array): Hurdle (0/1) for *all* buses.
            - hop_zone_idx_positive (np.array): Zone index for *positive* buses.
            - positive_durations (np.array): Duration values for *positive* buses.
            - n_zones (int): The number of unique zones created.
    """
    print("Preprocessing duration data...")
    if hop_col not in df.columns:
        raise ValueError(f"Missing required column: {hop_col}")
    if duration_col not in df.columns:
        raise ValueError(f"Missing required column: {duration_col}")

    df_proc = df.copy()

    # --- 1. Discretize Hop Distance ---
    df_proc['hop_distance_discrete'] = pd.cut(
        df_proc[hop_col], 
        bins=n_discrete, 
        labels=False, # Use integer labels 0 to n_discrete-1
        include_lowest=True,
        duplicates='drop'
    )
    
    if df_proc['hop_distance_discrete'].isnull().any():
        warnings.warn("NaNs found in 'hop_distance_discrete' after binning. Filling with zone 0.")
        df_proc['hop_distance_discrete'] = df_proc['hop_distance_discrete'].fillna(0)

    hop_zone_idx_all = pd.Categorical(df_proc['hop_distance_discrete']).codes.astype(int)
    n_zones = len(np.unique(hop_zone_idx_all))

    # --- 2. Create Hurdle Array (0s and 1s) ---
    is_positive_all = (df_proc[duration_col] > 0).astype(int)

    # --- 3. Filter for Positive Data ---
    positive_mask = is_positive_all == 1
    hop_zone_idx_positive = hop_zone_idx_all[positive_mask]
    positive_durations = df_proc.loc[positive_mask, duration_col].values

    print(f"Preprocessing complete. Found {len(hop_zone_idx_all)} total buses.")
    print(f"Found {len(positive_durations)} buses with positive duration.")
    
    return (hop_zone_idx_all, is_positive_all, 
            hop_zone_idx_positive, positive_durations, n_zones)


# ====================================================================
# --- Part 2: The Bayesian Duration Model Class ---
# ====================================================================


# ====================================================================
# --- Part 1: Data Preprocessing Utility ---
# ====================================================================

def preprocess_duration_data(df, n_discrete=10, 
                             hop_col='hop_distance_normalized', 
                             duration_col='duration'):
    """
    Preprocesses a raw DataFrame for the BayesianDurationModel.

    This function:
    1. Discretizes a 'hop_distance' column into zones.
    2. Creates the 'is_positive' (hurdle) array.
    3. Filters data for positive durations and their corresponding zones.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_discrete (int): Number of bins for hop distance.
        hop_col (str): Column name for normalized hop distance.
        duration_col (str): Column name for the duration data.

    Returns:
        tuple: A tuple containing:
            - hop_zone_idx_all (np.array): Zone index for *all* buses.
            - is_positive_all (np.array): Hurdle (0/1) for *all* buses.
            - hop_zone_idx_positive (np.array): Zone index for *positive* buses.
            - positive_durations (np.array): Duration values for *positive* buses.
            - n_zones (int): The number of unique zones created.
    """
    print("Preprocessing duration data...")
    if hop_col not in df.columns:
        raise ValueError(f"Missing required column: {hop_col}")
    if duration_col not in df.columns:
        raise ValueError(f"Missing required column: {duration_col}")

    df_proc = df.copy()

    # --- 1. Discretize Hop Distance ---
    df_proc['hop_distance_discrete'] = pd.cut(
        df_proc[hop_col], 
        bins=n_discrete, 
        labels=False, # Use integer labels 0 to n_discrete-1
        include_lowest=True,
        duplicates='drop'
    )
    
    if df_proc['hop_distance_discrete'].isnull().any():
        warnings.warn("NaNs found in 'hop_distance_discrete' after binning. Filling with zone 0.")
        df_proc['hop_distance_discrete'] = df_proc['hop_distance_discrete'].fillna(0)

    hop_zone_idx_all = pd.Categorical(df_proc['hop_distance_discrete']).codes.astype(int)
    n_zones = len(np.unique(hop_zone_idx_all))

    # --- 2. Create Hurdle Array (0s and 1s) ---
    is_positive_all = (df_proc[duration_col] > 0).astype(int)

    # --- 3. Filter for Positive Data ---
    positive_mask = is_positive_all == 1
    hop_zone_idx_positive = hop_zone_idx_all[positive_mask]
    positive_durations = df_proc.loc[positive_mask, duration_col].values

    print(f"Preprocessing complete. Found {len(hop_zone_idx_all)} total buses.")
    print(f"Found {len(positive_durations)} buses with positive duration.")
    
    return (hop_zone_idx_all, is_positive_all, 
            hop_zone_idx_positive, positive_durations, n_zones)


# ====================================================================
# --- Part 2: The Bayesian Duration Model Class ---
# ====================================================================

class BayesianDurationModel:
    """
    A class to learn and generate synthetic interruption durations (CAIDI)
    using a Hurdle-Weibull model based on hop distance.
    """
    
    def __init__(self, trace_path=None):
        """
        Initializes the Bayesian Duration Model.

        Args:
            trace_path (str, optional): Path to a pre-computed trace file
                                        (e.g., 'trace.nc').
        """
        self.model = None
        self.trace = None
        self.n_train_buses_all = None
        self.n_train_buses_pos = None
        self.n_zones = None

        if(trace_path==None):
            print("No trace_path provided. Fetching default pre-trained model...")
            trace_path = fetch_trace("duration")

        if trace_path:
            print(f"Loading trace from {trace_path}...")
            try:
                self.trace = az.from_netcdf(trace_path)
                
                # Extract key dimensions from the loaded trace
                self.n_train_buses_all = self.trace.observed_data.bus_all.shape[0]
                self.n_train_buses_pos = self.trace.observed_data.bus_positive.shape[0]
                self.n_zones = self.trace.posterior.hop_zone.shape[0]
                
                print(f"Successfully loaded pre-trained model.")
                print(f"Model was trained with {self.n_train_buses_all} total buses")
                print(f"({self.n_train_buses_pos} positive) and {self.n_zones} zones.")
            except Exception as e:
                warnings.warn(f"Could not load trace from {trace_path}. Error: {e}")
                print("You must call .learn() with data to train a new model.")
        else:
            print("No trace path provided. Call .learn() to train a new model.")

    def _build_model(self, n_buses_all, n_buses_pos, n_zones, 
                     hop_zone_all_data,           # Data for the Mutable container
                     hop_zone_pos_data,           # Data for the Mutable container
                     observed_is_positive,      # Likelihood data (or None)
                     observed_positive_durations): # Likelihood data (or None)
        """
        Internal method to define the PyMC model structure.
        """
        coords = {
            "bus_all": range(n_buses_all),
            "bus_positive": range(n_buses_pos),
            "hop_zone": range(n_zones)
        }

        with pm.Model(coords=coords) as hurdle_model:
            # --- Data Placeholders (ONLY for covariates) ---
            # These must be given valid, shaped data
            hop_zone_all_in = pm.MutableData("hop_zone_all_in", hop_zone_all_data, dims="bus_all")
            hop_zone_pos_in = pm.MutableData("hop_zone_pos_in", hop_zone_pos_data, dims="bus_positive")
            
            # ====================================================================
            # Part 1: The Hurdle (Zero vs. Positive)
            # ====================================================================
            p_positive_by_zone = pm.Beta("p_positive_by_zone", alpha=1.0, beta=1.0, dims="hop_zone")
            p_for_each_bus = p_positive_by_zone[hop_zone_all_in]

            observed_hurdle = pm.Bernoulli(
                "observed_hurdle",
                p=p_for_each_bus,
                observed=observed_is_positive, # <-- Use the arg, which can be None
                dims="bus_all"
            )

            # ====================================================================
            # Part 2: The Duration (Positive Values)
            # ====================================================================
            alpha_weibull_by_zone = pm.HalfNormal("alpha_weibull_by_zone", sigma=5, dims="hop_zone")
            beta_weibull_by_zone = pm.HalfNormal("beta_weibull_by_zone", sigma=10, dims="hop_zone")
            
            alpha_for_each_obs = alpha_weibull_by_zone[hop_zone_pos_in]
            beta_for_each_obs = beta_weibull_by_zone[hop_zone_pos_in]

            observed_positive_duration = pm.Weibull(
                "observed_positive_duration",
                alpha=alpha_for_each_obs, # shape
                beta=beta_for_each_obs,  # scale
                observed=observed_positive_durations, # <-- Use the arg, which can be None
                dims="bus_positive"
            )
        return hurdle_model
    
    def learn(self, hop_zone_idx_all, is_positive_all, 
              hop_zone_idx_positive, positive_durations, 
              n_zones=None, **sample_kwargs):
        """
        Trains the Bayesian Hurdle-Weibull model on new data.
        """
        self.n_train_buses_all = len(hop_zone_idx_all)
        self.n_train_buses_pos = len(positive_durations)
        if n_zones is None:
            self.n_zones = int(np.max(hop_zone_idx_all)) + 1
        else:
            self.n_zones = n_zones
        
        print(f"Building model for learning with {self.n_train_buses_all} total buses")
        print(f"({self.n_train_buses_pos} positive) and {self.n_zones} zones...")

        # Use max(1, ...) to avoid empty coordinates if no positive data exists
        n_buses_pos_safe = max(1, self.n_train_buses_pos)
        hop_zone_pos_safe = hop_zone_idx_positive if self.n_train_buses_pos > 0 else np.zeros(1, dtype=int)
        pos_durs_safe = positive_durations if self.n_train_buses_pos > 0 else np.zeros(1)


        self.model = self._build_model(
            n_buses_all=self.n_train_buses_all,
            n_buses_pos=n_buses_pos_safe,
            n_zones=self.n_zones,
            # Covariate data
            hop_zone_all_data=hop_zone_idx_all,
            hop_zone_pos_data=hop_zone_pos_safe,
            # Observed data
            observed_is_positive=is_positive_all,
            observed_positive_durations=pos_durs_safe
        )
        
        sample_args = {'draws': 1000, 'tune': 1000, 'cores': 4}
        sample_args.update(sample_kwargs)
        
        print(f"Starting sampling with args: {sample_args}")
        with self.model:
            self.trace = pm.sample(**sample_args)
        
        # If we had no positive data, drop the dummy coord from the trace
        if self.n_train_buses_pos == 0:
            self.trace.posterior = self.trace.posterior.drop_dims("bus_positive")
            self.trace.observed_data = self.trace.observed_data.drop_dims("bus_positive")

        print("Learning complete. Trace is stored in self.trace.")
        return self.trace

    def generate_data(self, new_hop_zone_idx=None, net=None, random_seed=None):
        """
        Generates synthetic interruption durations (CAIDI) for a new set of buses.
        """
        if self.trace is None:
            raise ValueError("No trace found. Please load a trace or run .learn() first.")
        
        if(new_hop_zone_idx==None):
            # --- 1. Calculate Hop Distance for all buses ---
            source_bus = net.ext_grid.bus.iloc[0] # This will be 0
            graph = pn.create_nxgraph(net)

            substation_node = net.bus.name.iloc[source_bus] # Get original osmnx node ID

            hop_distances_new_net = {}
            for bus_idx in graph.nodes:
                try:
                    dist = nx.shortest_path_length(graph, source=source_bus, target=bus_idx)
                    hop_distances_new_net[bus_idx] = dist
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    hop_distances_new_net[bus_idx] = np.nan

            hop_series = pd.Series(hop_distances_new_net, index=range(len(graph.nodes)))
            max_dist = hop_series.max()
            hop_series = hop_series.fillna(max_dist)
            print(f"Max hop distance found: {max_dist}")

            # --- 2. Discretize Bus Hop Distances for RELIABILITY (30 Bins) ---
            N_BINS_RELIABILITY = 30
            new_hop_zone_idx = pd.cut(
                hop_series, bins=N_BINS_RELIABILITY, labels=False, include_lowest=True
            ).values
            print(f"Created {len(new_hop_zone_idx)} bus zone indices for reliability (30 bins).")
        
        new_hop_zone_idx = np.asarray(new_hop_zone_idx)
        n_new_buses = len(new_hop_zone_idx)
        
        if np.max(new_hop_zone_idx) >= self.n_zones:
            warnings.warn(f"new_hop_zone_idx contains values >= {self.n_zones} (max trained zone). "
                          "Results may be unpredictable.")
        
        rng = np.random.default_rng(random_seed)

        # ====================================================================
        # --- Step 1: Generate Hurdle Predictions (Zero vs. Positive) ---
        # ====================================================================
        print(f"\n--- Part 1: Predicting the Hurdle for {n_new_buses} buses ---")
        
        # Use max(1, ...) to avoid empty coordinates
        n_buses_pos_safe = max(1, self.n_train_buses_pos)

        # Build a model instance with the *original* training dimensions
        # but NO observed data
        self.model = self._build_model(
            n_buses_all=self.n_train_buses_all,
            n_buses_pos=n_buses_pos_safe,
            n_zones=self.n_zones,
            # Placeholder covariate data (must have valid shape)
            hop_zone_all_data=np.zeros(self.n_train_buses_all, dtype=int),
            hop_zone_pos_data=np.zeros(n_buses_pos_safe, dtype=int),
            # No observed data for prediction
            observed_is_positive=None,
            observed_positive_durations=None
        )
        
        predicted_hurdles_xr = None
        
        with self.model:
            # We must also set data for hop_zone_pos_in, even if we don't
            # use it for hurdle prediction, just to keep the model graph valid.
            placeholder_pos_data = np.zeros(n_buses_pos_safe, dtype=int)

            if n_new_buses <= self.n_train_buses_all:
                print("New network is smaller or same size. Using padding...")
                
                padding_size = self.n_train_buses_all - n_new_buses
                padding_values = rng.integers(0, self.n_zones, size=padding_size)
                padded_hop_zone_idx = np.concatenate([new_hop_zone_idx, padding_values])
                
                pm.set_data({
                    "hop_zone_all_in": padded_hop_zone_idx,
                    "hop_zone_pos_in": placeholder_pos_data
                })
                
                ppc_hurdle_full = pm.sample_posterior_predictive(
                    self.trace,
                    var_names=["observed_hurdle"],
                    random_seed=random_seed,
                    extend_inferencedata=False
                )
                
                predicted_hurdles_xr = ppc_hurdle_full.posterior_predictive['observed_hurdle'].isel(
                    bus_all=slice(0, n_new_buses)
                )

            else:
                print("New network is larger. Generating hurdle predictions in chunks...")
                
                ppc_chunks = []
                for i in range(0, n_new_buses, self.n_train_buses_all):
                    chunk = new_hop_zone_idx[i : i + self.n_train_buses_all]
                    current_chunk_size = len(chunk)

                    if current_chunk_size < self.n_train_buses_all:
                        padding_size = self.n_train_buses_all - current_chunk_size
                        padding_values = rng.integers(0, self.n_zones, size=padding_size)
                        chunk = np.concatenate([chunk, padding_values])
                    
                    pm.set_data({
                        "hop_zone_all_in": chunk,
                        "hop_zone_pos_in": placeholder_pos_data
                    })
                    
                    ppc_chunk_full = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=["observed_hurdle"],
                        random_seed=random_seed,
                        extend_inferencedata=False
                    )
                    
                    ppc_chunks.append(
                        ppc_chunk_full.posterior_predictive["observed_hurdle"].isel(
                            bus_all=slice(0, current_chunk_size)
                        )
                    )
                
                predicted_hurdles_xr = xr.concat(ppc_chunks, dim="bus_all")

        # Convert to a (chain, draw, n_new_buses) numpy array
        predicted_hurdles = predicted_hurdles_xr.values

        # ====================================================================
        # --- Step 2: Generate Potential Positive Durations Manually ---
        # ====================================================================
        
        # (This part remains the same as your code)
        
        print("\n--- Part 2: Generating Potential Positive Durations ---")

        post_alpha = self.trace.posterior["alpha_weibull_by_zone"].values
        post_beta = self.trace.posterior["beta_weibull_by_zone"].values

        alpha_samples_for_new_buses = post_alpha[..., new_hop_zone_idx]
        beta_samples_for_new_buses = post_beta[..., new_hop_zone_idx]

        potential_durations = weibull_min.rvs(
            c=alpha_samples_for_new_buses,
            scale=beta_samples_for_new_buses,
            size=alpha_samples_for_new_buses.shape,
            random_state=rng
        )

        # ====================================================================
        # --- Step 3: Combine Hurdle and Duration Predictions ---
        # ====================================================================
        
        # (This part also remains the same)

        print("\n--- Part 3: Combining Hurdle and Duration Samples ---")

        final_dic_predictions = predicted_hurdles * potential_durations
        
        n_chains, n_draws, n_buses = final_dic_predictions.shape
        final_dic_predictions_flat = final_dic_predictions.reshape((n_chains * n_draws, n_buses))

        print("\n--- Simulation Complete ---")
        print(f"Shape of final samples: {final_dic_predictions_flat.shape}")
        
        return final_dic_predictions_flat
       
    def save_trace(self, file_path):
        """
        Saves the model's trace to a NetCDF file.

        Args:
            file_path (str): The destination file path (e.g., 'new_trace.nc').
        """
        if self.trace:
            self.trace.to_netcdf(file_path)
            print(f"Trace saved to {file_path}")
        else:
            print("No trace to save. Run .learn() first.")




# ====================================================================
# --- Part 1: Data Preprocessing Utility ---
# ====================================================================

def preprocess_frequency_data(df, n_discrete=10, 
                              hop_col='hop_distance_normalized', 
                              frequency_col='FIC_total'):
    """
    Preprocesses a raw DataFrame for the BayesianFrequencyModel.

    This function:
    1. Discretizes a 'hop_distance' column into zones.
    2. Extracts the frequency data (e.g., FIC_total).

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_discrete (int): Number of bins for hop distance.
        hop_col (str): Column name for normalized hop distance.
        frequency_col (str): Column name for the frequency count data.

    Returns:
        tuple: A tuple containing:
            - hop_zone_idx (np.array): Zone index for each bus.
            - frequency_data (np.array): Frequency value for each bus.
            - n_zones (int): The number of unique zones created.
    """
    print("Preprocessing frequency data...")
    if hop_col not in df.columns:
        raise ValueError(f"Missing required column: {hop_col}")
    if frequency_col not in df.columns:
        raise ValueError(f"Missing required column: {frequency_col}")

    df_proc = df.copy()

    # --- 1. Discretize Hop Distance ---
    df_proc['hop_distance_discrete'] = pd.cut(
        df_proc[hop_col], 
        bins=n_discrete, 
        labels=False, # Use integer labels 0 to n_discrete-1
        include_lowest=True,
        duplicates='drop'
    )
    
    if df_proc['hop_distance_discrete'].isnull().any():
        warnings.warn("NaNs found in 'hop_distance_discrete' after binning. Filling with zone 0.")
        df_proc['hop_distance_discrete'] = df_proc['hop_distance_discrete'].fillna(0)

    hop_zone_idx = pd.Categorical(df_proc['hop_distance_discrete']).codes.astype(int)
    n_zones = len(np.unique(hop_zone_idx))

    # --- 2. Get Frequency Data ---
    frequency_data = df_proc[frequency_col].values
    if np.isnan(frequency_data).any():
        warnings.warn("NaNs found in frequency data. Replacing with 0.")
        frequency_data = np.nan_to_num(frequency_data, nan=0.0)
    
    # Ensure data is integer for count model
    frequency_data = np.round(frequency_data).astype(int)

    print(f"Preprocessing complete. Found {len(hop_zone_idx)} buses across {n_zones} zones.")
    
    return hop_zone_idx, frequency_data, n_zones


# ====================================================================
# --- Part 2: The Bayesian Frequency Model Class ---
# ====================================================================

class BayesianFrequencyModel:
    """
    A class to learn and generate synthetic failure frequencies (CAIFI/FIC)
    using a Negative Binomial model based on hop distance.
    """
    
    def __init__(self, trace_path=None):
        """
        Initializes the Bayesian Frequency Model.

        Args:
            trace_path (str, optional): Path to a pre-computed trace file
                                        (e.g., 'trace.nc').
        """
        self.model = None
        self.trace = None
        self.n_train_buses = None
        self.n_zones = None

        if(trace_path==None):
            print("No trace_path provided. Fetching default pre-trained model...")
            trace_path = fetch_trace("frequency")

        if trace_path:
            print(f"Loading trace from {trace_path}...")
            try:
                self.trace = az.from_netcdf(trace_path)
                
                # Extract key dimensions from the loaded trace
                self.n_train_buses = self.trace.observed_data.bus.shape[0]
                self.n_zones = self.trace.posterior.hop_zone.shape[0]
                
                print(f"Successfully loaded pre-trained model.")
                print(f"Model was trained with {self.n_train_buses} buses and {self.n_zones} zones.")
            except Exception as e:
                warnings.warn(f"Could not load trace from {trace_path}. Error: {e}")
                print("You must call .learn() with data to train a new model.")
        else:
            print("No trace path provided. Call .learn() to train a new model.")

    def _build_model(self, n_buses, n_zones, 
                     hop_zone_idx_data,      # For MutableData
                     observed_frequency):    # For Likelihood (can be None)
        """
        Internal method to define the PyMC model structure.
        """
        coords = {
            "bus": range(n_buses),
            "hop_zone": range(n_zones)
        }

        with pm.Model(coords=coords) as fic_model:
            # --- Data Placeholder (Covariate) ---
            hop_zone_idx = pm.MutableData("hop_zone_idx", hop_zone_idx_data, dims="bus")

            # --- Priors for each Zone ---
            mu_by_zone = pm.HalfNormal("mu_by_zone", sigma=10, dims="hop_zone")
            
            # --- Overdispersion Parameter ---
            alpha_dispersion = pm.HalfNormal("alpha_dispersion", sigma=1)
            
            # --- Indexing ---
            mu_for_each_bus = mu_by_zone[hop_zone_idx]

            # --- Likelihood ---
            frequency_likelihood = pm.NegativeBinomial(
                "frequency_likelihood",
                mu=mu_for_each_bus,
                alpha=alpha_dispersion,
                observed=observed_frequency, # This can be None for prediction
                dims="bus"
            )
        return fic_model

    def learn(self, hop_zone_idx, frequency_data, n_zones=None, **sample_kwargs):
        """
        Trains the Bayesian Negative Binomial model on new data.

        Args:
            hop_zone_idx (np.array): Zone index for each bus.
            frequency_data (np.array): Frequency (count) for each bus.
            n_zones (int, optional): The total number of unique zones. 
                                    If None, inferred from hop_zone_idx.max() + 1.
            **sample_kwargs: Additional arguments passed to pm.sample() 
                             (e.g., draws=1000, tune=1000).
        """
        self.n_train_buses = len(hop_zone_idx)
        if n_zones is None:
            self.n_zones = int(np.max(hop_zone_idx)) + 1
        else:
            self.n_zones = n_zones
        
        print(f"Building model for learning with {self.n_train_buses} buses and {self.n_zones} zones...")

        self.model = self._build_model(
            n_buses=self.n_train_buses,
            n_zones=self.n_zones,
            hop_zone_idx_data=hop_zone_idx,
            observed_frequency=frequency_data
        )
        
        sample_args = {'draws': 1000, 'tune': 1000, 'cores': 4}
        sample_args.update(sample_kwargs)
        
        print(f"Starting sampling with args: {sample_args}")
        with self.model:
            self.trace = pm.sample(**sample_args)
        
        print("Learning complete. Trace is stored in self.trace.")
        return self.trace

    def generate_data(self, net=None, new_hop_zone_idx=None, random_seed=None):
        """
        Generates synthetic failure frequencies (CAIFI/FIC) for a new set of buses.

        Args:
            new_hop_zone_idx (list or np.array): A list/array of hop zone indices
                                                 for the new synthetic grid.
            random_seed (int, optional): Seed for reproducible generation.

        Returns:
            np.array: Shape (n_samples, n_new_buses).
                      The final sampled failure frequencies.
        """
        if self.trace is None:
            raise ValueError("No trace found. Please load a trace or run .learn() first.")
        
        if(new_hop_zone_idx==None):
            
            # Keep these for the next steps
            source_bus = net.ext_grid.bus.iloc[0] # This will be 0
            graph = pn.create_nxgraph(net)
            # --- 1. Calculate Hop Distance for all buses ---
            hop_distances_new_net = {}
            for bus_idx in graph.nodes:
                try:
                    dist = nx.shortest_path_length(graph, source=source_bus, target=bus_idx)
                    hop_distances_new_net[bus_idx] = dist
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    hop_distances_new_net[bus_idx] = np.nan

            hop_series = pd.Series(hop_distances_new_net, index=range(len(graph.nodes)))
            max_dist = hop_series.max()
            hop_series = hop_series.fillna(max_dist)
            print(f"Max hop distance found: {max_dist}")

            # --- 2. Discretize Bus Hop Distances for RELIABILITY (30 Bins) ---
            N_BINS_RELIABILITY = 30
            new_hop_zone_idx = pd.cut(
                hop_series, bins=N_BINS_RELIABILITY, labels=False, include_lowest=True
            ).values
            print(f"Created {len(new_hop_zone_idx)} bus zone indices for reliability (30 bins).")
                    
        new_hop_zone_idx = np.asarray(new_hop_zone_idx)
        n_new_buses = len(new_hop_zone_idx)
        
        if np.max(new_hop_zone_idx) >= self.n_zones:
            warnings.warn(f"new_hop_zone_idx contains values >= {self.n_zones} (max trained zone). "
                          "Results may be unpredictable.")
        
        # RNG for padding
        rng = np.random.default_rng(random_seed)

        # Build a model instance with the *original* training dimensions
        # but NO observed data
        self.model = self._build_model(
            n_buses=self.n_train_buses,
            n_zones=self.n_zones,
            hop_zone_idx_data=np.zeros(self.n_train_buses, dtype=int), # Placeholder
            observed_frequency=None # No observed data for prediction
        )
        
        predicted_frequencies_xr = None
        
        with self.model:
            if n_new_buses <= self.n_train_buses:
                # --- Case 1: New network is smaller or equal in size ---
                print("New network is smaller or same size. Using padding...")
                
                padding_size = self.n_train_buses - n_new_buses
                padding_values = rng.integers(0, self.n_zones, size=padding_size)
                padded_hop_zone_idx = np.concatenate([new_hop_zone_idx, padding_values])
                
                pm.set_data({"hop_zone_idx": padded_hop_zone_idx})
                
                ppc_full = pm.sample_posterior_predictive(
                    self.trace,
                    var_names=["frequency_likelihood"],
                    random_seed=random_seed,
                    extend_inferencedata=False
                )
                
                predicted_frequencies_xr = ppc_full.posterior_predictive["frequency_likelihood"].isel(
                    bus=slice(0, n_new_buses)
                )

            else:
                # --- Case 2: New network is larger ---
                print("New network is larger. Generating predictions in chunks...")
                
                ppc_chunks = []
                for i in range(0, n_new_buses, self.n_train_buses):
                    chunk = new_hop_zone_idx[i : i + self.n_train_buses]
                    current_chunk_size = len(chunk)

                    if current_chunk_size < self.n_train_buses:
                        padding_size = self.n_train_buses - current_chunk_size
                        padding_values = rng.integers(0, self.n_zones, size=padding_size)
                        chunk = np.concatenate([chunk, padding_values])
                    
                    pm.set_data({"hop_zone_idx": chunk})
                    
                    ppc_chunk_full = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=["frequency_likelihood"],
                        random_seed=random_seed,
                        extend_inferencedata=False
                    )
                    
                    ppc_chunks.append(
                        ppc_chunk_full.posterior_predictive["frequency_likelihood"].isel(
                            bus=slice(0, current_chunk_size)
                        )
                    )
                
                predicted_frequencies_xr = xr.concat(ppc_chunks, dim="bus")

        # --- Format and return the output ---
        # Convert to a (chain, draw, n_new_buses) numpy array
        predicted_frequencies = predicted_frequencies_xr.values
        
        # Stack chains and draws into a single 'sample' dimension
        n_chains, n_draws, n_buses = predicted_frequencies.shape
        final_fic_flat = predicted_frequencies.reshape((n_chains * n_draws, n_buses))

        print("\n--- Simulation Complete ---")
        print(f"Shape of final samples: {final_fic_flat.shape}")
        
        return final_fic_flat

    def save_trace(self, file_path):
        """
        Saves the model's trace to a NetCDF file.

        Args:
            file_path (str): The destination file path (e.g., 'new_trace.nc').
        """
        if self.trace:
            self.trace.to_netcdf(file_path)
            print(f"Trace saved to {file_path}")
        else:
            print("No trace to save. Run .learn() first.")




# ====================================================================
# --- Part 1: Data Preprocessing Utility ---
# ====================================================================

def preprocess_impedance_data(df, n_discrete=10, 
                              elec_dist_col='electrical_distance_to_substation', 
                              r1_col='R1', x1_col='X1'):
    """
    Preprocesses a raw DataFrame for the BayesianImpedanceModel.

    This function:
    1. Discretizes an 'electrical_distance' column into zones.
    2. Extracts R1 and X1 data.
    3. Handles and filters NaN values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_discrete (int): Number of bins for electrical distance.
        elec_dist_col (str): Column name for electrical distance.
        r1_col (str): Column name for R1 (resistance).
        x1_col (str): Column name for X1 (reactance).

    Returns:
        tuple: A tuple containing:
            - elec_dist_idx (np.array): Zone index for each line.
            - r1_data (np.array): R1 value for each line.
            - x1_data (np.array): X1 value for each line.
            - n_zones (int): The number of unique zones created.
    """
    print("Preprocessing impedance data...")
    if elec_dist_col not in df.columns:
        raise ValueError(f"Missing required column: {elec_dist_col}")

    df_proc = df.copy()

    # --- 1. Discretize Electrical Distance ---
    df_proc['elec_dist_discrete'] = pd.cut(
        df_proc[elec_dist_col], 
        bins=n_discrete, 
        labels=False, # Use integer labels 0 to n_discrete-1
        include_lowest=True,
        duplicates='drop'
    )
    
    if df_proc['elec_dist_discrete'].isnull().any():
        warnings.warn("NaNs found in 'elec_dist_discrete' after binning. Filling with zone 0.")
        df_proc['elec_dist_discrete'] = df_proc['elec_dist_discrete'].fillna(0)

    elec_dist_idx_data = df_proc['elec_dist_discrete'].values
    r1_data = df_proc[r1_col].values
    x1_data = df_proc[x1_col].values
    
    # --- 2. Filter NaNs ---
    valid_mask = (~np.isnan(r1_data) & 
                  ~np.isnan(x1_data) & 
                  ~np.isnan(elec_dist_idx_data))
    
    r1_data = r1_data[valid_mask]
    x1_data = x1_data[valid_mask]
    elec_dist_idx = elec_dist_idx_data[valid_mask].astype(int)

    n_zones = len(np.unique(elec_dist_idx))

    print(f"Preprocessing complete. Found {len(r1_data)} valid lines across {n_zones} zones.")
    
    return elec_dist_idx, r1_data, x1_data, n_zones


# ====================================================================
# --- Part 2: The Bayesian Impedance Model Class ---
# ====================================================================

class BayesianImpedanceModel:
    """
    A class to learn and generate synthetic line impedance (R1 and X1)
    using a Gamma Mixture Model based on electrical distance.
    """
    
    def __init__(self, trace_r_path=None, trace_x_path=None, n_mixture_components=3):
        """
        Initializes the Bayesian Impedance Model.

        Args:
            trace_r_path (str): Path to the pre-computed R1 trace file (.pickle).
            trace_x_path (str): Path to the pre-computed X1 trace file (.pickle).
            n_mixture_components (int): The number of mixture components in the models.
        """
        self.r_model_gen = None
        self.x_model_gen = None
        self.trace_r = None
        self.trace_x = None
        self.n_train_lines_r = None
        self.n_zones_r = None
        self.n_train_lines_x = None
        self.n_zones_x = None
        self.n_components = n_mixture_components

        if trace_r_path is None:
            print("Fetching default R-model trace...")
            trace_r_path = fetch_trace("impedance_r")
            
        if trace_x_path is None:
            print("Fetching default X-model trace...")
            trace_x_path = fetch_trace("impedance_x")

        # --- Load R1 Trace ---
        try:
            print(f"Loading R1 trace from {trace_r_path}...")
            self.trace_r =  az.from_netcdf(trace_r_path)
            # Infer dimensions from the R trace
            self.n_train_lines_r = self.trace_r['observed_data']['line_segment'].shape[0]
            self.n_zones_r = max(self.trace_r['constant_data']['elec_dist_idx'].values)+1
            print(f"R1 model loaded. Trained with {self.n_train_lines_r} lines and {self.n_zones_r} zones.")
        except Exception as e:
            warnings.warn(f"Could not load R1 trace from {trace_r_path}. Error: {e}")

        # --- Load X1 Trace ---
        try:
            print(f"Loading X1 trace from {trace_x_path}...")
            self.trace_x =  az.from_netcdf(trace_x_path)
            
            # Infer dimensions from the X trace
            self.n_train_lines_x = self.trace_x['observed_data']['line_segment'].shape[0]
            self.n_zones_x = max(self.trace_x['constant_data']['elec_dist_idx'].values)+1
            print(f"X1 model loaded. Trained with {self.n_train_lines_x} lines and {self.n_zones_x} zones.")
        except Exception as e:
            warnings.warn(f"Could not load X1 trace from {trace_x_path}. Error: {e}")

    def _build_model(self, n_lines, n_zones, elec_dist_data, 
                     observed_data, likelihood_name):
        """
        Internal method to define the PyMC model structure.
        """
        coords = {
            "line_segment": range(n_lines),
            "component": range(self.n_components),
            "elec_dist_zone": range(n_zones)
        }

        with pm.Model(coords=coords) as mixture_model:
            # --- Data Placeholder (Covariate) ---
            elec_dist_idx = pm.MutableData("elec_dist_idx", elec_dist_data, dims="line_segment")

            # --- Priors for Mixture Components (Global) ---
            mean_1 = pm.HalfNormal("mean_1", sigma=1)
            mean_2_offset = pm.HalfNormal("mean_2_offset", sigma=1)
            mean_3_offset = pm.HalfNormal("mean_3_offset", sigma=1)
            
            means = pm.Deterministic("means", pt.stack([
                mean_1, 
                mean_1 + mean_2_offset, 
                mean_1 + mean_2_offset + mean_3_offset
            ]), dims="component")

            cv = pm.HalfNormal("cv", sigma=0.5)
            sigmas = pm.Deterministic("sigmas", means * cv, dims="component")
            
            alphas = (means / sigmas)**2
            betas = means / sigmas**2

            # --- Priors for Mixture Weights (Zone-Dependent) ---
            w_by_zone = pm.Dirichlet(
                "w_by_zone", 
                a=np.ones(self.n_components), 
                dims=("elec_dist_zone", "component")
            )
            
            w_for_each_line = w_by_zone[elec_dist_idx]

            # --- Likelihood ---
            gamma_dists = pm.Gamma.dist(alpha=alphas, beta=betas)

            # Use the likelihood_name passed as argument
            likelihood = pm.Mixture(
                likelihood_name,
                w=w_for_each_line,
                comp_dists=gamma_dists,
                observed=observed_data, # Can be None for prediction
                dims="line_segment"
            )
        return mixture_model

    def _run_generation(self, trace, n_train_lines, n_zones, 
                        new_elec_dist_idx, likelihood_name, rng):
        """
        Internal helper to run the padding/chunking generation logic.
        """
        n_new_lines = len(new_elec_dist_idx)
        
        # Build the model for generation
        placeholder_data = np.zeros(n_train_lines, dtype=int)
        gen_model = self._build_model(
            n_lines=n_train_lines,
            n_zones=n_zones,
            elec_dist_data=placeholder_data,
            observed_data=None, # No observed data for prediction
            likelihood_name=likelihood_name
        )
        
        predicted_samples = None
        
        with gen_model:
            if n_new_lines <= n_train_lines:
                # --- Case 1: New network is smaller or equal ---
                padding_size = n_train_lines - n_new_lines
                padding_values = rng.integers(0, n_zones, size=padding_size)
                padded_idx = np.concatenate([new_elec_dist_idx, padding_values]).astype(int)
                
                pm.set_data({"elec_dist_idx": padded_idx})
                
                ppc_full = pm.sample_posterior_predictive(
                    trace,
                    var_names=[likelihood_name],
                    extend_inferencedata=False
                )
                
                predicted_samples = ppc_full.posterior_predictive[likelihood_name].values[..., :n_new_lines]

            else:
                # --- Case 2: New network is larger ---
                ppc_chunks = []
                for i in range(0, n_new_lines, n_train_lines):
                    chunk = new_elec_dist_idx[i : i + n_train_lines]
                    current_chunk_size = len(chunk)

                    if current_chunk_size < n_train_lines:
                        padding_size = n_train_lines - current_chunk_size
                        padding_values = rng.integers(0, n_zones, size=padding_size)
                        chunk = np.concatenate([chunk, padding_values])
                    
                    chunk = chunk.astype(int)
                    pm.set_data({"elec_dist_idx": chunk})
                    
                    ppc_chunk_full = pm.sample_posterior_predictive(
                        trace,
                        var_names=[likelihood_name],
                        extend_inferencedata=False
                    )
                    
                    ppc_chunks.append(
                        ppc_chunk_full.posterior_predictive[likelihood_name].values[..., :current_chunk_size]
                    )
                
                predicted_samples = np.concatenate(ppc_chunks, axis=2)
        
        return predicted_samples


    def generate_data(self, new_elec_dist_idx=None, net=None, random_seed=None):
        """
        Generates synthetic R1 and X1 values for a new set of lines.

        Args:
            new_elec_dist_idx (list or np.array): A list/array of electrical
                                                 distance zone indices for the
                                                 new synthetic lines.
            random_seed (int, optional): Seed for reproducible generation.

        Returns:
            tuple: (generated_r1, generated_x1)
                - generated_r1 (np.array): Shape (n_samples, n_new_lines)
                - generated_x1 (np.array): Shape (n_samples, n_new_lines)
        """
        if self.trace_r is None or self.trace_x is None:
            raise ValueError("Both R and X traces must be loaded to generate data.")
        
        if(self.n_train_lines_r==None or self.n_zones_r==None):
            self.n_train_lines_r = self.trace_r['observed_data']['line_segment'].shape[0]
            self.n_zones_r = max(self.trace_r['constant_data']['elec_dist_idx'].values)+1

        if(self.n_train_lines_x==None or self.n_zones_x==None):
            self.n_train_lines_x = self.trace_x['observed_data']['line_segment'].shape[0]
            self.n_zones_x = max(self.trace_x['constant_data']['elec_dist_idx'].values)+1

        if(new_elec_dist_idx==None):
            print("Calculating hop distances for all buses in the OSM network...")
            # Keep these for the next steps
            source_bus = net.ext_grid.bus.iloc[0] # This will be 0
            graph = pn.create_nxgraph(net)
            
            # --- 1. Calculate Hop Distance for all buses ---
            hop_distances_new_net = {}
            for bus_idx in net.bus.index:
                try:
                    dist = nx.shortest_path_length(graph, source=source_bus, target=bus_idx)
                    hop_distances_new_net[bus_idx] = dist
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    hop_distances_new_net[bus_idx] = np.nan

            hop_series = pd.Series(hop_distances_new_net, index=net.bus.index)
            max_dist = hop_series.max()
            hop_series = hop_series.fillna(max_dist)
            print(f"Max hop distance found: {max_dist}")

            # --- 2. Discretize Bus Hop Distances for POWER (10 Bins) ---
            N_BINS_POWER_IMPD = 10
            hop_zone_series_power = pd.cut(
                hop_series, bins=N_BINS_POWER_IMPD, labels=False, include_lowest=True
            )
            hop_zone_idx_power = hop_zone_series_power.values
            print(f"Created {len(hop_zone_idx_power)} bus zone indices for power (10 bins).")

            # --- 3. Discretize LINE Electrical Distances (10 Bins) ---
            from_buses = net.line.from_bus
            new_elec_dist_idx = hop_zone_series_power.loc[from_buses].values
            print(f"Created {len(new_elec_dist_idx)} line zone indices for impedance (10 bins).")
        
        new_elec_dist_idx = np.asarray(new_elec_dist_idx)
        n_new_lines = len(new_elec_dist_idx)
        rng = np.random.default_rng(random_seed)

        # --- Part 1: Generate R1 Samples ---
        print(f"Generating R1 samples for {n_new_lines} lines...")
        generated_r1_raw = self._run_generation(
            trace=self.trace_r,
            n_train_lines=self.n_train_lines_r,
            n_zones=self.n_zones_r,
            new_elec_dist_idx=new_elec_dist_idx,
            likelihood_name="r1_likelihood",
            rng=rng
        )

        # --- Part 2: Generate X1 Samples ---
        print(f"Generating X1 samples for {n_new_lines} lines...")
        generated_x1_raw = self._run_generation(
            trace=self.trace_x,
            n_train_lines=self.n_train_lines_x,
            n_zones=self.n_zones_x,
            new_elec_dist_idx=new_elec_dist_idx,
            likelihood_name="x1_likelihood",
            rng=rng
        )

        # --- Part 3: Format and Return ---
        n_chains_r, n_draws_r, _ = generated_r1_raw.shape
        generated_r1 = generated_r1_raw.reshape((n_chains_r * n_draws_r, n_new_lines))
        
        n_chains_x, n_draws_x, _ = generated_x1_raw.shape
        generated_x1 = generated_x1_raw.reshape((n_chains_x * n_draws_x, n_new_lines))

        print("\n--- Simulation Complete ---")
        print(f"R1 samples shape: {generated_r1.shape}")
        print(f"X1 samples shape: {generated_x1.shape}")

        return generated_r1, generated_x1
    
    
    def _learn(self, elec_dist_idx, data, n_zones, likelihood_name, **sample_kwargs):
        """Internal helper to run the learning process."""
        n_lines = len(elec_dist_idx)
        
        print(f"Building {likelihood_name} model for learning with {n_lines} lines...")

        model = self._build_model(
            n_lines=n_lines,
            n_zones=n_zones,
            elec_dist_data=elec_dist_idx,
            observed_data=data,
            likelihood_name=likelihood_name
        )
        
        sample_args = {'draws': 1000, 'tune': 1000, 'cores': 4}
        sample_args.update(sample_kwargs)
        
        trace = None
        print(f"Starting sampling with args: {sample_args}")
        with model:
            trace = pm.sample(**sample_args)
        
        print("Learning complete.")
        return model, trace

    def learn_r(self, elec_dist_idx, r1_data, n_zones, **sample_kwargs):
        """Trains the R1 model on new data."""
        model, trace = self._learn(elec_dist_idx, r1_data, n_zones, 
                                   "r1_likelihood", **sample_kwargs)
        self.r_model_gen = model
        self.trace_r = trace
        return trace
        
    def learn_x(self, elec_dist_idx, x1_data, n_zones, **sample_kwargs):
        """Trains the X1 model on new data."""
        model, trace = self._learn(elec_dist_idx, x1_data, n_zones, 
                                   "x1_likelihood", **sample_kwargs)
        self.x_model_gen = model
        self.trace_x = trace
        return trace
    
    def save_trace(self, trace, file_path):
        """
        Saves the model's trace to a NetCDF file.

        Args:
            file_path (str): The destination file path (e.g., 'new_trace.nc').
        """
        if trace:
            trace.to_netcdf(file_path)
            print(f"Trace saved to {file_path}")
        else:
            print("No trace to save. Run .learn() first.")














################ OPEN STREET MAP FUNCTIONS
def get_osm_graph(query, query_type='city', dist=1000, network_type='drive'):
    """
    Fetches a raw NetworkX graph from OpenStreetMap using various query types.

    Args:
        query (str or tuple or dict): The search query.
            - query_type='city': A city/place name (e.g., "São Carlos, Brazil").
            - query_type='point': A (lat, lon) tuple (e.g., (-23.6, -46.5)).
            - query_type='address': A full address string.
            - query_type='bbox': A dict {'north': Y, 'south': Y, 'east': X, 'west': X}.
        query_type (str): One of 'city', 'point', 'address', 'bbox'.
        dist (int): Distance in meters (used for 'point' and 'address' queries).
        network_type (str): OSM network type (e.g., 'drive', 'walk', 'all').

    Returns:
        networkx.MultiDiGraph: The raw, undirected graph from OSMNx.
    """
    print(f"Fetching graph for query '{query}' (type: {query_type})...")
    if query_type == 'city':
        G = ox.graph_from_place(query, network_type=network_type, simplify=True)
    elif query_type == 'point':
        G = ox.graph_from_point(query, dist=dist, network_type=network_type, simplify=True)
    elif query_type == 'address':
        G = ox.graph_from_address(query, dist=dist, network_type=network_type, simplify=True)
    elif query_type == 'bbox':
            bbox_tuple = (
                query['west'], 
                query['south'], 
                query['east'], 
                query['north']
            )
            G = ox.graph_from_bbox(
                bbox=bbox_tuple,
                network_type=network_type, 
                simplify=True
            )
    else:
        raise ValueError("Invalid query_type. Must be 'city', 'point', 'address', or 'bbox'.")
    
    # Ensure graph is undirected for topology analysis
    return nx.to_undirected(G)

def find_substation_node(G, substation_point=None):
    """
    Finds the nearest graph node to a (lat, lon) point.
    If no point is given, finds the node closest to the graph's center.
    """
    if substation_point:
        print(f"Finding nearest node to specified point {substation_point}...")
        substation_lon, substation_lat = substation_point[1], substation_point[0]
    else:
        print("No substation_point provided. Finding graph center...")
        
        # --- (FIXED LINE) ---
        # ox.utils_graph.graph_to_gdfs is deprecated.
        # We now use ox.graph_to_gdfs() which returns (gdf_nodes, gdf_edges)
        nodes_df, _ = ox.graph_to_gdfs(G)
        # --- (END FIX) ---
        
        substation_lon = nodes_df['x'].mean()
        substation_lat = nodes_df['y'].mean()
        print(f"Using graph center: ({substation_lat:.4f}, {substation_lon:.4f})")
        
    # Find the single nearest node
    substation_node = ox.distance.nearest_nodes(G, X=substation_lon, Y=substation_lat)
    print(f"Substation node selected: {substation_node}")
    return substation_node

def radialize_graph(G, source_node):
    """
    Converts a (potentially meshed) graph into a radial tree graph
    using a Breadth-First Search (BFS) from the source_node.
    """
    print(f"Radializing graph from source node {source_node}...")
    
    # Get the simple (u, v) edges that form the BFS tree
    bfs_edges_no_keys = list(nx.bfs_edges(G, source=source_node))
    
    # Reconstruct the full edge identifiers (u, v, key=0)
    # This assumes the primary edge (key=0) is the one we want.
    bfs_edges_with_keys = [(u, v, 0) for u, v in bfs_edges_no_keys]
    
    # Create the subgraph using the correct edge identifiers
    tree_graph = G.edge_subgraph(bfs_edges_with_keys).copy()
    
    print(f"Original graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Radial tree: {len(tree_graph.nodes())} nodes, {len(tree_graph.edges())} edges")
    return tree_graph

def convert_nx_to_pandapower(tree_graph, substation_node, line_std_type="NAYY 4x50 SE"):
    """
    Converts the final NetworkX tree graph into a pandapower network.
    The substation_node is created as bus 0 and the external grid.
    """
    print("Converting NetworkX graph to pandapower network...")
    net = pp.create_empty_network(name="OSM Synthetic Network")
    nx_to_pp_bus = {}

    # 1. Create the substation bus first (it will be index 0)
    sub_data = tree_graph.nodes[substation_node]
    bus_0_idx = pp.create_bus(
        net, vn_kv=13.8, name=str(substation_node), geodata=(sub_data['x'], sub_data['y'])
    )
    nx_to_pp_bus[substation_node] = bus_0_idx
    
    # 2. Create all other buses
    for node, data in tree_graph.nodes(data=True):
        if node != substation_node:
            bus_idx = pp.create_bus(
                net, vn_kv=13.8, name=str(node), geodata=(data['x'], data['y'])
            )
            nx_to_pp_bus[node] = bus_idx
            
    print(f"Created {len(net.bus)} buses.")

    # 3. Create lines from graph edges
    for u, v, data in tree_graph.edges(data=True):
        from_bus = nx_to_pp_bus[u]
        to_bus = nx_to_pp_bus[v]
        
        # OSMNx edge lengths are in meters, pandapower requires km
        length_km = data.get('length', 100) / 1000.0 # Default to 100m if no length
        
        pp.create_line(
            net, from_bus=from_bus, to_bus=to_bus, length_km=length_km,
            std_type=line_std_type, name=f"Line {u}-{v}"
        )
        
    print(f"Created {len(net.line)} lines.")

    # 4. Add the External Grid (Slack Bus) at bus 0
    pp.create_ext_grid(net, bus=bus_0_idx, vm_pu=1.0, name="Grid Connection")
    print(f"Added External Grid at bus 0 (Node {substation_node}).")
    
    return net

def create_osm_pandapower_network(query, query_type='city', dist=1000, 
                                substation_point=None, line_std_type="NAYY 4x50 SE"):
    """
    Main wrapper function. Fetches, processes, and converts an OSM
    road network into a radial pandapower network.
    """
    # 1. Get raw graph from OSM
    G_raw = get_osm_graph(query, query_type, dist)
    
    # 2. Find the substation node
    sub_node = find_substation_node(G_raw, substation_point)
    
    # 3. Create a radial tree from that node
    G_tree = radialize_graph(G_raw, sub_node)
    
    # 4. Convert the tree to a pandapower network
    net = convert_nx_to_pandapower(G_tree, sub_node, line_std_type)
    
    print("\n--- OSM pandapower network creation complete! ---")
    return net, G_tree




def save_power_phase_samples(gen_phases, gen_power, bus_index, phase_map, n_samples, output_path):
    """
    Processes and saves all power and phase samples to a CSV file.
    """
    print("Processing Power & Phase...")
    
    # Get dimensions
    n_buses = len(bus_index)
    
    # Create index columns
    sample_col = np.repeat(np.arange(n_samples), n_buses)
    bus_id_col = np.tile(bus_index.values, n_samples)
    
    # Flatten the data
    power_flat = gen_power.reshape(-1, 3) # (n_samples*n_buses, 3)
    phases_flat = gen_phases.flatten()     # (n_samples*n_buses)
    phases_str_col = [phase_map[i] for i in phases_flat]

    # Create and save the DataFrame
    df = pd.DataFrame({
        'sample_id': sample_col,
        'bus_id': bus_id_col,
        'P_A': power_flat[:, 0],
        'P_B': power_flat[:, 1],
        'P_C': power_flat[:, 2],
        'phase': phases_str_col
    })
    df.to_csv(output_path, index=False)
    print(f"Saved '{os.path.basename(output_path)}' with {len(df)} rows.")

def save_bus_metric_samples(gen_data, col_name, bus_index, n_samples, output_path):
    """
    Processes and saves a generic bus-level metric (like Freq/Dur) to a CSV.
    """
    print(f"Processing {col_name}...")
    
    n_buses = len(bus_index)
    sample_col = np.repeat(np.arange(n_samples), n_buses)
    bus_id_col = np.tile(bus_index.values, n_samples)
    data_flat = gen_data.flatten()
    
    df = pd.DataFrame({
        'sample_id': sample_col,
        'bus_id': bus_id_col,
        col_name: data_flat
    })
    df.to_csv(output_path, index=False)
    print(f"Saved '{os.path.basename(output_path)}' with {len(df)} rows.")

def save_impedance_samples(gen_r, gen_x, line_index, n_samples, output_path):
    """
    Processes and saves all R1 and X1 impedance samples to a CSV file.
    """
    print("Processing Impedance...")
    
    n_lines = len(line_index)
    sample_col = np.repeat(np.arange(n_samples), n_lines)
    line_id_col = np.tile(line_index.values, n_samples)
    r_flat = gen_r.flatten()
    x_flat = gen_x.flatten()

    df = pd.DataFrame({
        'sample_id': sample_col,
        'line_id': line_id_col,
        'R1_ohm_per_km': r_flat,
        'X1_ohm_per_km': x_flat
    })
    df.to_csv(output_path, index=False)
    print(f"Saved '{os.path.basename(output_path)}' with {len(df)} rows.")
    
print("Helper functions for saving are defined.")




