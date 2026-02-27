import osmnx as ox
import pandapower as pp
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import osmnx as ox
import numpy as np

import json
from matplotlib.patches import Patch
import re
import pandas as pd
import warnings
import pandapower as pp
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from scipy.spatial import cKDTree

def reset_net_index(single_net):
    # This function creates a continuous bus index starting at 0
    # and automatically updates all connected elements (lines, trafos, loads, etc.)
    pp.create_continuous_bus_index(single_net, start=0)
    
    # Reset the component indices (lines, trafos, etc) 
    # create_continuous_bus_index only fixes the BUS index, not the element indices
    for elm in pp.pp_elements(bus=False):
        if elm in single_net and not single_net[elm].empty:
            single_net[elm].reset_index(drop=True, inplace=True)
            
    return single_net


my_custom_tags = ox.settings.useful_tags_way + ['voltage'] 
ox.settings.useful_tags_way=my_custom_tags

ox.settings.useful_tags_node = ox.settings.useful_tags_node + ['voltage']

def get_centroid_from_geom(geom):
    if isinstance(geom, (Polygon, MultiPolygon)):
        return (geom.centroid.y, geom.centroid.x)
    elif isinstance(geom, Point):
        return (geom.y, geom.x)
    elif isinstance(geom, LineString):
        return (geom.centroid.y, geom.centroid.x)
    return (geom.centroid.y, geom.centroid.x)



def prune_irrelevant_transmission_branches(G, sub_coords, gen_coords):
    """
    Iteratively removes 'dead-end' branches from the graph that do not lead 
    to a Substation or a Generator.
    
    Args:
        G: NetworkX graph (Undirected)
        sub_coords: List of (lat, lon) for substations
        gen_coords: List of (lat, lon) for generators
    """
    G = G.copy()
    print(f"  > Pre-filter: {len(G.nodes)} nodes, {len(G.edges)} lines.")
    
    if len(G.nodes) == 0:
        return G

    # 1. Identify "Terminal Nodes" on the graph
    # These are the nodes closest to our physical assets (Subs + Gens)
    # We MUST keep these nodes and the paths between them.
    
    g_nodes = list(G.nodes)
    # G.nodes data is x,y (lon, lat) usually in OSMnx
    g_coords = np.array([[G.nodes[n]['y'], G.nodes[n]['x']] for n in g_nodes])
    
    if len(g_coords) == 0: return G
    
    tree = cKDTree(g_coords)
    
    # Map Substations to Graph Nodes
    terminals = set()
    if sub_coords:
        _, idxs = tree.query(sub_coords)
        for i in idxs:
            terminals.add(g_nodes[i])
            
    # Map Generators to Graph Nodes
    if gen_coords:
        _, idxs = tree.query(gen_coords)
        for i in idxs:
            terminals.add(g_nodes[i])
            
    print(f"  > Identified {len(terminals)} terminal nodes (Grid connections).")

    # 2. Iterative Pruning (The "Haircut" Algorithm)
    # Repeatedly remove nodes that have Degree=1 (leaves) AND are not Terminals.
    # This eats away dead-ends until it hits a Terminal or a Loop.
    
    removed_count = 0
    while True:
        # Find leaves that are NOT important terminals
        nodes_to_remove = [n for n in G.nodes 
                           if G.degree(n) == 1 and n not in terminals]
        
        if not nodes_to_remove:
            break # Converged
            
        G.remove_nodes_from(nodes_to_remove)
        removed_count += len(nodes_to_remove)

    # 3. Component Cleanup
    # Remove floating islands that might contain loops but NO terminals
    # (e.g. a disconnected ring of transmission lines in the forest)
    components = list(nx.connected_components(G))
    for comp in components:
        # If this island has NO terminal nodes, it serves no purpose
        if not set(comp).intersection(terminals):
            G.remove_nodes_from(comp)
            removed_count += len(comp)

    print(f"  > Post-filter: {len(G.nodes)} nodes. Removed {removed_count} irrelevant nodes.")
    return G


def plot_multi_feeder_graph(G_forest, sub_nodes, 
                            sub_color='black', sub_size=300, 
                            node_size_base=20, 
                            cmap_name='tab20',
                            to_legend=False):
    """
    Visualizes the multi-substation network using OSMnx.
    
    - Substations are large and black (or custom color).
    - Each radial feeder gets a unique color from the colormap.
    """
    
    print("--- Preparing Plot Data ---")
    
    # 1. Identify Connected Components (The separate feeders)
    # Each component corresponds to one Voronoi cluster/tree
    components = list(nx.connected_components(G_forest))
    print(f"Identified {len(components)} distinct electrical islands (feeders).")
    
    # 2. Generate a Color Map
    # We need as many colors as there are feeders
    cmap = plt.get_cmap(cmap_name)
    # Create a map: Component_Index -> Color
    # We use a modulo operator just in case there are more feeders than colors in the map
    comp_color_map = {i: mcolors.to_hex(cmap(i % cmap.N)) for i in range(len(components))}
    
    # Map Node -> Color based on which component it belongs to
    node_color_dict = {}
    for i, comp_nodes in enumerate(components):
        color = comp_color_map[i]
        for node in comp_nodes:
            node_color_dict[node] = color

    # 3. Build the Lists for OSMnx (Must match G.nodes() order)
    node_colors = []
    node_sizes = []
    
    # Convert sub_nodes to set for O(1) lookup
    sub_nodes_set = set(sub_nodes)
    
    for node in G_forest.nodes():
        # A. Check if node is a Substation
        if node in sub_nodes_set:
            c = node_color_dict.get(node, '#cccccc') 
            node_colors.append('black') # Substations are distinct
            node_sizes.append(sub_size)
            
        # B. Regular Node
        else:
            # Get the color of the feeder it belongs to
            # Default to gray if something went wrong and it's isolated
            c = node_color_dict.get(node, '#cccccc') 
            node_colors.append(c)
            node_sizes.append(node_size_base)

    # 4. Plot using OSMnx
    print("Plotting graph...")
    fig, ax = ox.plot_graph(
        G_forest,
        node_color=node_colors,
        node_size=node_sizes,
        node_zorder=2,
        edge_color='#333333',     # Dark grey lines
        edge_linewidth=1.0,
        edge_alpha=0.8,
        bgcolor='white',
        show=False,
        close=False,
        figsize=(12, 12)
    )
    
    # Optional: Add a dummy legend for "Substation"
    # (Matplotlib tricks to add custom legend elements)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Substation',
               markerfacecolor=sub_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='LV Node',
               markerfacecolor='gray', markersize=5),
    ]
    if(to_legend==True):
        ax.legend(handles=legend_elements, loc='upper right')
    
    #plt.show()
    return fig, ax





def plot_detailed_integrated_system(net,draw_transformers=True,min_kv=0,show_legend=True,to_save=False):
    """
    Plots the full integrated system with specific Voltage Classes:
    - High Voltage (>= 69 kV): Red, Thick
    - Medium Voltage (1 kV - 69 kV): Blue, Medium
    - Low Voltage (< 1 kV): Black, Thin
    """
    
    # --- 1. Setup & Helper Functions ---
    plt.figure(figsize=(16, 14),dpi=300)
    ax = plt.gca()
    
    # Helper to extract coordinates
    def get_coords(net):
        coords = {}
        for i in net.bus.index:
            if i not in coords:
                d = json.loads(net.bus.at[i, 'geo'])
                coords[i] = (d['coordinates'][0], d['coordinates'][1])
        return coords

    bus_coords = get_coords(net)
    
    # --- Voltage Classification Logic ---
    def get_voltage_style(vn_kv):
        if vn_kv >= 69.0:
            # High Voltage (>= 69 kV)
            return {'color': "#000000", 'width': 12.0, 'zorder': 3, 'label': 'High Voltage (≥ 69 kV)'}
        elif vn_kv >= 1.0:
            # Medium Voltage (1 kV - 69 kV)
            return {'color': "#00FF00", 'width': 3.0, 'zorder': 2, 'label': 'Medium Voltage (1-69 kV)'}
        else:
            # Low Voltage (< 1 kV)
            return {'color': "#FF0000", 'width': 3.0, 'zorder': 1, 'label': 'Low Voltage (< 1 kV)'}

    # --- 2. Draw Lines ---
    print("Drawing Lines...")
    for idx, line in net.line.iterrows():
        try:
            f_bus = line.from_bus
            t_bus = line.to_bus
            
            if f_bus in bus_coords and t_bus in bus_coords:
                p1 = bus_coords[f_bus]
                p2 = bus_coords[t_bus]
                
                # Determine Voltage (use max nominal voltage of connected buses)
                vn_kv = max(net.bus.at[f_bus, 'vn_kv'], net.bus.at[t_bus, 'vn_kv'])
                
                style = get_voltage_style(vn_kv)
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color=style['color'], 
                        linewidth=style['width'], 
                        zorder=style['zorder'], 
                        alpha=0.8)
        except: pass

    # --- 3. Draw Transformers ---
    print("Drawing Transformers...")
    if(draw_transformers):
        for idx, trafo in net.trafo.iterrows():
            try:
                hv, lv = trafo.hv_bus, trafo.lv_bus
                lv_level = trafo.vn_lv_kv
                if hv in bus_coords and lv in bus_coords and lv_level>min_kv:
                    p1, p2 = bus_coords[hv], bus_coords[lv]
                    mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

                    
                    # Connector
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=2)
                    # Symbol
                    ax.scatter(mid_x, mid_y, s=400, c='purple', marker='d', edgecolors='black', zorder=4)
            except: pass

    # # --- 4. Draw Generators & Sources ---
    # print("Drawing Assets...")
    # # Generators
    # for idx, gen in net.sgen.iterrows():
    #     if gen.bus in bus_coords:
    #         x, y = bus_coords[gen.bus]
    #         ax.scatter(x, y, s=200, c='#00FF00', marker='^', edgecolors='black', zorder=5)

    # External Grids (PCC)
    for idx, ext in net.ext_grid.iterrows():
        if ext.bus in bus_coords:
            x, y = bus_coords[ext.bus]
            ax.scatter(x, y, s=250, c='red', marker='s', edgecolors='black', zorder=6)

    # --- 5. Formatting & Legend ---
    ax.set_aspect('equal')
    ax.axis('off')
    #plt.title("Integrated Power System Topology\nBy Voltage Class", fontsize=16)

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color="#000000", lw=12.0, label='High Voltage (≥ 69 kV)'),
        Line2D([0], [0], color='#00FF00', lw=12.0, label='Medium Voltage (1-69 kV)'),
        Line2D([0], [0], color='#FF0000', lw=12.0, label='Low Voltage (< 1 kV)'),
        Line2D([0], [0], color='purple', lw=2, linestyle=':', marker='d', markersize=16, label='Transformer'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='red', markeredgecolor='k', markersize=16, label='Grid Source'),
    ]
    if(show_legend):
        ax.legend(handles=legend_elements, fontsize=30, framealpha=0.9, ncols=1)
    plt.tight_layout()
    if(to_save):
        plt.savefig('integrated_network.svg', format='svg', dpi=300)
    plt.show()





# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def parse_voltage_tag(tag):
    if not tag: return 138
    tag_clean = str(tag).split(';')[0].split(',')[0]
    nums = re.findall(r'\d+', tag_clean)
    if not nums: return 138
    val = float(nums[0])
    return val / 1000.0 if val > 500 else max(val, 69)


def get_centroid(geom):
    if isinstance(geom, Point): return geom.x, geom.y
    elif isinstance(geom, (Polygon, MultiPolygon)): return geom.centroid.x, geom.centroid.y
    return None, None

def parse_voltage_list(tag):
    """
    Parses OSM voltage tag into a LIST of floats.
    Input: "230000;88000;13800" -> Output: [230.0, 88.0, 13.8]
    """
    if pd.isna(tag) or not tag: 
        return [13.8] # Default if missing
    
    tag_str = str(tag).replace(',', '.') 
    parts = tag_str.split(';')
    
    valid_volts = []
    for p in parts:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", p)
        if nums:
            val = float(nums[0])
            # Normalize to kV
            if val > 1000: val /= 1000.0
            valid_volts.append(val)
            
    if not valid_volts:
        return [13.8]
    
    # Return unique sorted values (descending order usually helpful, but we treat them as set)
    return sorted(list(set(valid_volts)), reverse=True)

# ==========================================
# 2. OSM GRAPH & SUBSTATION FETCHING
# ==========================================

def get_osm_graph(query, query_type='city', dist=1000, network_type='drive'):
    print(f"Fetching graph for query '{query}'...")
    if query_type == 'city': G = ox.graph_from_place(query, network_type=network_type, simplify=True)
    elif query_type == 'point': G = ox.graph_from_point(query, dist=dist, network_type=network_type, simplify=True)
    elif query_type == 'address': G = ox.graph_from_address(query, dist=dist, network_type=network_type, simplify=True)
    elif query_type == 'bbox': G = ox.graph_from_bbox(query['north'], query['south'], query['east'], query['west'], network_type=network_type)
    return nx.to_undirected(G)


def get_osm_substations(query, query_type='city', dist=1000, 
                        transmission_connected=False, 
                        transmission_radius=5000, 
                        connection_threshold=0):
    """
    Fetches substations. 
    Updated for OSMnx 2.0+ (replaced project_gdf with to_crs).
    """
    
    search_dist = transmission_radius if transmission_connected else dist
    print(f"Searching for substations (Transmission Mode: {transmission_connected}, Radius: {search_dist}m)...")
    
    tags_sub = {'power': 'substation'}
    tags_line = {'power': 'line'} 

    try:
        # 1. Fetch Substations
        if query_type == 'city':
            gdf_subs = ox.features_from_place(query, tags_sub)
        elif query_type == 'point':
            gdf_subs = ox.features_from_point(query, tags_sub, dist=search_dist)
        elif query_type == 'address':
            gdf_subs = ox.features_from_address(query, tags_sub, dist=search_dist)
        elif query_type == 'bbox':
            gdf_subs = ox.features_from_bbox((query['north'], query['south'], query['east'], query['west']), tags_sub)
        else:
            return []
            
        if gdf_subs.empty:
            print("No substations found in the area.")
            return []

        # 2. Transmission Connectivity Filter
        if transmission_connected:
            print("Verifying connection to transmission grid...")
            
            try:
                if query_type == 'city':
                    gdf_lines = ox.features_from_place(query, tags_line)
                elif query_type in ['point', 'address']:
                    gdf_lines = ox.features_from_point(query, tags_line, dist=search_dist)
                elif query_type == 'bbox':
                    gdf_lines = ox.features_from_bbox((query['north'], query['south'], query['east'], query['west']), tags_line)
                else:
                    gdf_lines = None
            except Exception as e:
                print(f"Warning fetching lines: {e}")
                gdf_lines = None

            if gdf_lines is None or gdf_lines.empty:
                print("Warning: No transmission lines found. Cannot verify connection. Returning empty.")
                return []

            # --- FIX FOR OSMNX 2.0+ ---
            # Use GeoPandas standard projection estimation
            try:
                # 1. Estimate UTM CRS for metric calculation
                utm_crs = gdf_subs.estimate_utm_crs()
                
                # 2. Project both GDFs to that CRS
                gdf_subs_proj = gdf_subs.to_crs(utm_crs)
                gdf_lines_proj = gdf_lines.to_crs(utm_crs)
            except Exception as e:
                print(f"Projection failed: {e}. Skipping distance check.")
                gdf_subs_proj = gdf_subs
                gdf_lines_proj = gdf_lines

            # Filter Substations
            lines_union = gdf_lines_proj.unary_union
            connected_indices = []
            
            for idx, row in gdf_subs_proj.iterrows():
                dist_to_grid = row.geometry.distance(lines_union)
                if dist_to_grid <= connection_threshold:
                    connected_indices.append(idx)
            
            initial_count = len(gdf_subs)
            gdf_subs = gdf_subs.loc[connected_indices]
            print(f"Filtered {initial_count} -> {len(gdf_subs)} substations connected to grid.")
            
            if gdf_subs.empty:
                return []

        # 3. Parse Voltage & Geometry
        substations_data = []
        
        if 'voltage' not in gdf_subs.columns:
            gdf_subs['voltage'] = np.nan
            
        for idx, row in gdf_subs.iterrows():
            x, y = get_centroid(row.geometry)
            if x is not None:
                raw_v = row.get('voltage', None)
                voltages = parse_voltage_list(raw_v)
                
                for v in voltages:
                    substations_data.append({
                        'osm_id': idx,       
                        'coords': (y, x),    
                        'voltage_kv': v      
                    })
                
        return substations_data

    except Exception as e:
        print(f"Warning: Could not fetch substations: {e}")
        return []
# ==========================================
# 1. CONSTANTS & HELPERS
# ==========================================

STANDARD_VOLTAGES = np.array([69, 115, 138, 161, 230, 345, 500])

TRANS_LINE_PARAMS = {
    500: {'x_pu_km': 0.000155, 'x_r': 17.0},
    345: {'x_pu_km': 0.000360, 'x_r': 16.0},
    230: {'x_pu_km': 0.000945, 'x_r': 12.5},
    161: {'x_pu_km': 0.001828, 'x_r': 10.0},
    138: {'x_pu_km': 0.002471, 'x_r': 9.1},
    115: {'x_pu_km': 0.003398, 'x_r': 8.3},
    69:  {'x_pu_km': 0.003398, 'x_r': 8.3}
}

TRAFO_PARAMS = {
    69:  {'mva': 42,  'x_r': 20},
    115: {'mva': 53,  'x_r': 25},
    138: {'mva': 83,  'x_r': 30},
    161: {'mva': 100, 'x_r': 32},
    230: {'mva': 203, 'x_r': 44},
    345: {'mva': 444, 'x_r': 60},
    500: {'mva': 812, 'x_r': 70}
}

def get_nearest_voltage(v_input):
    if pd.isna(v_input): return 138
    idx = (np.abs(STANDARD_VOLTAGES - v_input)).argmin()
    return STANDARD_VOLTAGES[idx]

def parse_voltage_tag(tag):
    if not tag: return 138
    tag_clean = str(tag).split(';')[0].split(',')[0]
    nums = re.findall(r'\d+', tag_clean)
    if not nums: return 138
    val = float(nums[0])
    return val / 1000.0 if val > 500 else max(val, 69)
def calculate_line_params(voltage_kv):
    params = TRANS_LINE_PARAMS.get(get_nearest_voltage(voltage_kv), TRANS_LINE_PARAMS[138])
    x_pu_km = params['x_pu_km']
    x_r_ratio = params['x_r']
    z_base = (voltage_kv ** 2) / 100.0
    x_ohm_km = x_pu_km * z_base
    r_ohm_km = x_ohm_km / x_r_ratio
    #print('line params',voltage_kv,params)
    return r_ohm_km, x_ohm_km

def calculate_trafo_params(hv_kv):
    params = TRAFO_PARAMS.get(get_nearest_voltage(hv_kv), TRAFO_PARAMS[138])
    sn_mva = params['mva']
    x_r = params['x_r']
    vk_percent = 12.0
    vkr_percent = vk_percent / np.sqrt(1 + x_r**2)
    #print('trafo params',hv_kv,params)
    return sn_mva, vkr_percent, vk_percent

# ==========================================
# 3. TOPOLOGICAL PARTITIONING
# ==========================================

def partition_and_radialize(G_raw, sub_nodes_list):
    print(f"Partitioning network for {len(sub_nodes_list)} substations (Topological Method)...")
    
    if not nx.is_connected(G_raw):
        largest_cc = max(nx.connected_components(G_raw), key=len)
        G_raw = G_raw.subgraph(largest_cc).copy()
    
    valid_sub_nodes = [s for s in sub_nodes_list if s in G_raw.nodes]
    
    if not valid_sub_nodes:
        fallback = sorted(G_raw.degree, key=lambda x: x[1], reverse=True)[0][0]
        valid_sub_nodes = [fallback]

    G_temp = G_raw.copy()
    super_source = "SUPER_SOURCE_VIRTUAL_NODE"
    G_temp.add_node(super_source)
    for sub in valid_sub_nodes:
        G_temp.add_edge(super_source, sub, length=0, weight=0)
        
    try:
        predecessors = nx.shortest_path(G_temp, source=super_source, weight='length')
    except: return nx.MultiGraph(), []

    G_forest = nx.MultiGraph()
    G_forest.graph = G_raw.graph.copy()
    G_forest.add_nodes_from(G_raw.nodes(data=True))
    
    for node in G_raw.nodes():
        if node == super_source or node not in predecessors: continue
        path = predecessors[node]
        if len(path) < 3: continue 
        
        parent = path[-2]
        edge_data = G_raw.get_edge_data(parent, node)
        best_attr = min(edge_data.values(), key=lambda x: x.get('length', 1.0))
        G_forest.add_edge(parent, node, key=0, **best_attr)

    return G_forest, valid_sub_nodes

def convert_forest_to_pandapower(G_tree, sub_node_map, line_std_type):
    """
    sub_node_map: Dict {osm_node_id: [list_of_voltages]}
    """
    net = pp.create_empty_network()
    net.bus['geo'] = None
    
    node_to_bus = {}
    components = list(nx.connected_components(G_tree))
    
    # Map every node to the voltage of its feeder source
    node_voltage_map = {}
    
    for comp in components:
        # Find roots
        roots = list(set(comp).intersection(set(sub_node_map.keys())))
        
        if roots:
            root = roots[0]
            # If a substation has multiple voltages (e.g. 138 and 13.8), 
            # and it feeds a distribution grid, we assume the grid operates 
            # at the LOWEST MV voltage available at that node.
            # The higher voltages are reserved for transmission connections.
            available_voltages = sub_node_map[root]
            #feeder_voltage = min(available_voltages)
            feeder_voltage = 13.8
        else:
            feeder_voltage = 13.8
            
        for node in comp:
            node_voltage_map[node] = feeder_voltage

    # Create Buses
    for n, data in G_tree.nodes(data=True):
        vn_kv = node_voltage_map.get(n, 13.8)
        b_idx = pp.create_bus(net, vn_kv=vn_kv, name=str(n), geodata=(data['y'], data['x']))
        net.bus.at[b_idx, 'geo'] = json.dumps({"type": "Point", "coordinates": [data['x'], data['y']]})
        node_to_bus[n] = b_idx
        
    # Create External Grids (Sources)
    for sub_osm_id in sub_node_map.keys():
        if sub_osm_id in node_to_bus:
            pp.create_ext_grid(net, bus=node_to_bus[sub_osm_id], name=f"Sub_{sub_osm_id}")
            
    # Create Lines
    for u, v, data in G_tree.edges(data=True):
        if u in node_to_bus and v in node_to_bus:
            l_km = data.get('length', 100) / 1000.0
            pp.create_line(net, from_bus=node_to_bus[u], to_bus=node_to_bus[v],
                           length_km=max(0.01, l_km), std_type=line_std_type)
    return net



def create_multi_feeder_network(query, query_type='city', dist=1000, line_std_type="NAYY 4x50 SE",
                                transmission_connected=False): 
    if(transmission_connected):
        print("Note: Transmission-connected substations enabled. This may take longer to fetch data.")
        G_raw = get_osm_graph(query, query_type, dist)
        sub_data = get_osm_substations(query, query_type, dist, transmission_connected=transmission_connected,
                                    transmission_radius=dist)
    else:
        G_raw = get_osm_graph(query, query_type, dist)
        sub_data = get_osm_substations(query, query_type, dist)

    #print('oii', len(sub_data))
    
    # Map: OSM_Node_ID -> List of Voltages
    sub_node_map = {}
    
    if not sub_data:
        print("No substations found. Using graph center (13.8 kV).")
        lats = [G_raw.nodes[n]['y'] for n in G_raw.nodes]
        lons = [G_raw.nodes[n]['x'] for n in G_raw.nodes]
        center_node = ox.nearest_nodes(G_raw, np.mean(lons), np.mean(lats))
        sub_node_map[center_node] = [13.8]
    else:
        lats = [d['coords'][0] for d in sub_data]
        lons = [d['coords'][1] for d in sub_data]
        # Map physical locations to Street Nodes
        osm_ids = ox.nearest_nodes(G_raw, lons, lats)
        
        for oid, entry in zip(osm_ids, sub_data):
            v = entry['voltage_kv']
            if oid not in sub_node_map:
                sub_node_map[oid] = []
            sub_node_map[oid].append(v)

    # Clean duplicates in voltage lists
    for k in sub_node_map:
        sub_node_map[k] = sorted(list(set(sub_node_map[k])))

    # Partition
    sub_nodes_list = list(sub_node_map.keys())
    G_forest, valid_subs = partition_and_radialize(G_raw, sub_nodes_list)
    
    valid_sub_map = {k: v for k, v in sub_node_map.items() if k in valid_subs}

    # Convert to Global Pandapower Net
    net = convert_forest_to_pandapower(G_forest, valid_sub_map, line_std_type)
    
    # --- NEW: Extract Individual Feeder Networks ---
    print("Extracting individual feeder networks...")
    single_feeder_net_list = []
    
    # We rely on the fact that each feeder is electrically isolated at this stage (before Trans integration)
    # We iterate over every external grid (substation)
    for ext_grid_idx in net.ext_grid.index:
        try:
            # Get the bus connected to this ext_grid
            root_bus = net.ext_grid.at[ext_grid_idx, 'bus']
            
            # Find all buses connected to this root (topology search)
            # pp.topology.connected_component returns a set of buses connected to the start bus
            graph_from_net = pp.topology.create_nxgraph(net)
            feeder_buses = pp.topology.connected_component(graph_from_net, root_bus)
            
            # Create a new net with ONLY these buses and associated elements

            feeder_net = pp.select_subnet(net, buses=feeder_buses, include_results=False)
            
            # Ensure the feeder_net has the 'geo' column copied correctly (select_subnet might skip custom cols)
            # We re-populate it from original net
            feeder_net.bus['geo'] = None
            # Map original indices to new indices to copy data if needed, 
            # but select_subnet usually preserves names.
            # Pandapower reindexes buses from 0..N in the new net.
            # We rely on 'name' to map back if needed, but select_subnet keeps data usually.
            
            # Manually copy 'geo' column if missing after select_subnet
            if 'geo' not in feeder_net.bus.columns:
                 feeder_net.bus['geo'] = None

            # Iterate to copy geo based on matching names (safest way)
            # net.bus is indexed by integer, feeder_net.bus is indexed by integer.
            # We match via 'name' which is the OSM ID str
            orig_geo_map = dict(zip(net.bus['name'], net.bus['geo']))
            
            for i in feeder_net.bus.index:
                name = feeder_net.bus.at[i, 'name']
                if name in orig_geo_map:
                    feeder_net.bus.at[i, 'geo'] = orig_geo_map[name]

            single_feeder_net_list.append(feeder_net)
            
        except Exception as e:
            print(f"Warning: Failed to extract feeder for ext_grid {ext_grid_idx}: {e}")

    print(f"--- Network Created: {len(valid_sub_map)} Feeders. Extracted {len(single_feeder_net_list)} single nets. ---")
    
    return net, G_forest, valid_sub_map, single_feeder_net_list




def apply_voltage_levels(net, tree_graph, sub_node_map, lv_kv=0.127):
    """
    Modifies the network to have dual voltage levels (MV/LV).
    Works for both global combined networks and single feeder networks.
    """
    print("--- Applying Dual-Voltage Levels (Dynamic Source Voltage) ---")

    # 1. Identify Connected Components (Feeders)
    # If tree_graph is directed (DiGraph), connected_components won't work directly, need to_undirected()
    if tree_graph.is_directed():
        undirected_graph = tree_graph.to_undirected()
        components = list(nx.connected_components(undirected_graph))
    else:
        components = list(nx.connected_components(tree_graph))

    sub_set = set(sub_node_map.keys())
    lv_nodes = set()
    
    # Store the detected MV level for each component to use during transformer creation
    feeder_mv_map = {} # {osm_node_id: mv_voltage}

    for comp in components:
        # Find the root (substation) of this feeder
        # We look for which node in this component is in our list of known substations
        roots = list(sub_set.intersection(comp))
        
        if not roots: 
            # If no root found in component, check if the net has a single Ext_Grid 
            # and infer root from there (fallback for single feeder nets passed without full sub_map)
            # But assuming correct input usage:
            continue
        
        root = roots[0]
        
        # DETERMINE MV LEVEL FOR THIS FEEDER
        # Logic: We use the lowest available voltage at the sub as the MV distribution level
        if root in sub_node_map:
            feeder_mv = min(sub_node_map[root])
        else:
            feeder_mv = 13.8 # Fallback
        
        # Map all nodes in this component to this MV level (for lookup later)
        for n in comp:
            feeder_mv_map[n] = feeder_mv
            
        # Identify LV Leaves
        # Degree 1, but NOT the substation itself
        # Note: tree_graph degree logic works best on Undirected graphs for leaf detection
        leaves = [n for n in comp if tree_graph.degree(n) == 1 and n not in sub_set]
        
        for leaf in leaves:
            try:
                path = nx.shortest_path(tree_graph, root, leaf)
                # Reverse to walk Leaf -> Source
                path_from_leaf = path[::-1]
                
                for node in path_from_leaf:
                    # Stop if we hit source
                    if node in sub_set: 
                        break
                    
                    # Stop if we hit junction (Degree > 2)
                    if tree_graph.degree(node) > 2: 
                        break
                    
                    # Else, it's LV
                    lv_nodes.add(node)
            except: continue

    # 2. Update Bus Voltages in Pandapower
    net.bus['name'] = net.bus['name'].astype(str)
    name_to_idx = dict(zip(net.bus['name'], net.bus.index))
    
    for n in lv_nodes:
        if str(n) in name_to_idx: 
            net.bus.at[name_to_idx[str(n)], 'vn_kv'] = lv_kv

    # 3. Process Junctions and Insert Transformers
    junction_map = {} # MV_Bus_Index -> List of Line Indices
    
    for idx, line in net.line.iterrows():
        fb = line.from_bus
        tb = line.to_bus
        v_from = net.bus.at[fb, 'vn_kv']
        v_to = net.bus.at[tb, 'vn_kv']
        
        if v_from != v_to:
            # Found a boundary (MV to LV)
            # Determine which bus is the MV Junction (High Voltage side of this specific line)
            mv_bus = fb if v_from > v_to else tb
            
            if mv_bus not in junction_map:
                junction_map[mv_bus] = []
            junction_map[mv_bus].append(idx)

    # Perform Node Splitting
    trafos_created = 0
    for mv_bus_idx, lines in junction_map.items():
        # Retrieve the correct MV voltage for *this specific location*
        local_mv_kv = net.bus.at[mv_bus_idx, 'vn_kv']
        
        # Get Geo
        try:
            geo_raw = net.bus.at[mv_bus_idx, 'geo']
            if geo_raw:
                coords = json.loads(geo_raw)['coordinates']
                geo_tuple = (coords[1], coords[0]) # (lat, lon)
                geo_json = geo_raw
            else:
                geo_tuple = (0,0)
                geo_json = "{}"
        except:
            geo_tuple = (0,0)
            geo_json = "{}"
            
        # Create new LV bus at same location
        lv_bus_idx = pp.create_bus(net, vn_kv=lv_kv, name=f"LV_Split_{mv_bus_idx}", geodata=geo_tuple)
        net.bus.at[lv_bus_idx, 'geo'] = geo_json
        
        # Move lines to new LV bus
        for l_idx in lines:
            if net.line.at[l_idx, 'from_bus'] == mv_bus_idx:
                net.line.at[l_idx, 'from_bus'] = lv_bus_idx
            else:
                net.line.at[l_idx, 'to_bus'] = lv_bus_idx
            
            # Set std type for LV
            net.line.at[l_idx, 'std_type'] = 'NAYY 4x50 SE'
            
        # Create Transformer
        pp.create_transformer_from_parameters(
            net, hv_bus=mv_bus_idx, lv_bus=lv_bus_idx,
            sn_mva=0.63, 
            vn_hv_kv=local_mv_kv, vn_lv_kv=lv_kv,
            vkr_percent=1.0, vk_percent=4.0, pfe_kw=0.5, i0_percent=0.1,
            name=f"Trafo_Split_{mv_bus_idx}"
        )
        trafos_created += 1

    # Fix MV Line Types
    for idx, line in net.line.iterrows():
        # If both sides are > 1kV (approx for MV check)
        v1 = net.bus.at[line.from_bus, 'vn_kv']
        v2 = net.bus.at[line.to_bus, 'vn_kv']
        if v1 > 1.0 and v2 > 1.0:
            net.line.at[idx, 'std_type'] = 'NAYY 4x150 SE'

    print(f"Topology Updated: {trafos_created} transformers inserted.")
    return net



def build_transmission_generation_network(dist_net, dist_radius=10000, 
                                          line_filter='["power"~"line|tower|substation"]',
                                          query_point=None):
    print("--- Starting Integrated Grid Parameterization (Multi-Source + Auto-Trafo) ---")
    
    # ======================================================
    # 1. SETUP COORDINATES (CENTROID OF DISTRIBUTION GRID)
    # ======================================================
    
    dist_sub_indices = dist_net.ext_grid.bus.tolist()
    
    if not dist_sub_indices:
        print("Error: No external grids found in distribution network.")
        return dist_net, nx.Graph()

    lats = []
    lons = []
    
    for bus_idx in dist_sub_indices:
        try:
            geo_str = dist_net.bus.at[bus_idx, 'geo']
            coords = json.loads(geo_str)['coordinates'] # [lon, lat]
            lons.append(coords[0])
            lats.append(coords[1])
        except Exception as e:
            print(f"Warning: Could not parse geo for bus {bus_idx}: {e}")
            continue

    if not lats:
        print("Error: No coordinates found for substations.")
        return dist_net, nx.Graph()
        
    if(query_point==None):
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        query_point = (center_lat, center_lon)

    # ======================================================
    # 2. FETCH OSM DATA
    # ======================================================
    print(f"Querying Transmission OSM data around {query_point}...")
    
    # A. Transmission Lines
    try:
        G_trans = ox.graph_from_point(query_point, dist=dist_radius, 
                                      custom_filter=line_filter, simplify=True)
        G_trans = nx.to_undirected(G_trans)
    except:
        G_trans = nx.Graph()
        print("Warning: No transmission lines found.")

    # B. Generators
    generators_data = [] 
    try:
        gens_df = ox.features_from_point(query_point, {'power': 'generator'}, dist=dist_radius)
        if not gens_df.empty:
            for idx, row in gens_df.iterrows():
                generators_data.append({
                    'id': f"gen_{idx}", 
                    'pos': get_centroid_from_geom(row.geometry), 
                    'source': row.get('generator:source', 'other')
                })
    except Exception as e:
        print(f"Gen fetch error: {e}")
        pass

    # ======================================================
    # NEW STEP: FILTER GRAPH
    # ======================================================
    
    # Prepare lists of coordinates for the filter function
    # sub_lats/lons come from Step 1 of your existing function
    current_sub_coords = list(zip(lats, lons)) 
    current_gen_coords = [g['pos'] for g in generators_data]
    
    # Apply the Filter
    if len(G_trans.nodes) > 0:
        G_trans = prune_irrelevant_transmission_branches(
            G_trans, 
            sub_coords=current_sub_coords, 
            gen_coords=current_gen_coords
        )

    # ======================================================
    # 3. BUILD PANDAPOWER NETWORK
    # ======================================================
    
    net = dist_net 
    
    osm_to_pp_bus = {}
    trans_coords_list = []
    trans_node_ids = []
    
    # A. Build the Transmission Backbone (Nodes)
    if len(G_trans.nodes) > 0:
        for n, data in G_trans.nodes(data=True):
            
            node_voltages = []
            for u, v, d in G_trans.edges(n, data=True):
                v_tag = parse_voltage_tag(d.get('voltage', None))
                if v_tag: node_voltages.append(v_tag)
            
            # If we found voltages on connected lines, take the max (or mode)
            # Default to 138kV if unknown
            bus_kv = max(node_voltages) if node_voltages else 138.0
            
            b_idx = pp.create_bus(net, vn_kv=bus_kv, name=f"Trans_{n}",
                                  geodata=(data['y'], data['x']))
            
            net.bus.at[b_idx, 'geo'] = json.dumps({"type": "Point", "coordinates": [data['x'], data['y']]})
            
            osm_to_pp_bus[n] = b_idx
            trans_coords_list.append((data['y'], data['x']))
            trans_node_ids.append(n)

        # B. Build Transmission Edges (Lines OR Transformers)
        teste = True
        for u, v, data in G_trans.edges(data=True):
             if u in osm_to_pp_bus and v in osm_to_pp_bus:
                bus_u = osm_to_pp_bus[u]
                bus_v = osm_to_pp_bus[v]
                
                v_u = net.bus.at[bus_u, 'vn_kv']
                v_v = net.bus.at[bus_v, 'vn_kv']

                
                length_km = data.get('length', 500) / 1000.0
                
                #print(v_u,v_v)
                # --- CHECK VOLTAGE MISMATCH ---
                if abs(v_u - v_v) > 1.0: 
                    #print('transformador transmissao')
                    # Voltages differ > 1kV -> Insert TRANSFORMER
                    
                    hv_bus = bus_u if v_u > v_v else bus_v
                    lv_bus = bus_v if v_u > v_v else bus_u
                    vn_hv = max(v_u, v_v)
                    vn_lv = min(v_u, v_v)
                    
                    # Calculate Trafo Params based on HV side
                    sn_mva, vkr, vk = calculate_trafo_params(vn_hv)
                    
                    pp.create_transformer_from_parameters(
                        net, hv_bus=hv_bus, lv_bus=lv_bus,
                        sn_mva=sn_mva, vn_hv_kv=vn_hv, vn_lv_kv=vn_lv,
                        vkr_percent=vkr, vk_percent=vk, pfe_kw=0, i0_percent=0,
                        name=f"Trans_Interconnect_{u}_{v}"
                    )
                    # print(f"  > Created Trafo between {vn_hv}kV and {vn_lv}kV nodes")

                else:
                    # Voltages are same -> Insert LINE
                    # Use the bus voltage for param calculation
                    #print('oi!')
                    r_ohm, x_ohm = calculate_line_params(v_u)
                    
                    pp.create_line_from_parameters(
                        net, from_bus=bus_u, to_bus=bus_v,
                        length_km=max(0.1, length_km), 
                        r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm, c_nf_per_km=10, max_i_ka=2.0,
                        name=f"TransLine_{u}_{v}"
                    )

    # C. Connect EACH Distribution Substation to the Grid
    if trans_coords_list:
        trans_tree = cKDTree(trans_coords_list)
        
        # Remove old distribution slacks (they become loads/trafos)
        net.ext_grid.drop(net.ext_grid.index, inplace=True)
        
        for i, dist_bus_idx in enumerate(dist_sub_indices):
            try:
                coords = json.loads(net.bus.at[dist_bus_idx, 'geo'])['coordinates'] # [lon, lat]
                sub_lon, sub_lat = coords[0], coords[1]
                
                # Find Nearest Transmission Node
                _, idx = trans_tree.query((sub_lat, sub_lon))
                closest_trans_node_osm = trans_node_ids[idx]
                closest_trans_bus_pp = osm_to_pp_bus[closest_trans_node_osm]
                
                trans_kv = net.bus.at[closest_trans_bus_pp, 'vn_kv']
                dist_kv = net.bus.at[dist_bus_idx, 'vn_kv']
                
                # 1. Create HV Interface Bus AT the Substation
                sub_hv_bus = pp.create_bus(net, vn_kv=trans_kv, name=f"Sub_HV_Input_{i}",
                                           geodata=(sub_lat, sub_lon))
                net.bus.at[sub_hv_bus, 'geo'] = json.dumps({"type": "Point", "coordinates": [sub_lon, sub_lat]})
                
                # 2. Transmission Line (Grid -> Substation)
                line_dist_km = ox.distance.great_circle(sub_lat, sub_lon, 
                                                        trans_coords_list[idx][0], trans_coords_list[idx][1]) / 1000.0
                
                r_line, x_line = calculate_line_params(trans_kv)
                
                pp.create_line_from_parameters(
                    net, from_bus=closest_trans_bus_pp, to_bus=sub_hv_bus,
                    length_km=max(0.1, line_dist_km),
                    r_ohm_per_km=r_line, x_ohm_per_km=x_line, c_nf_per_km=9, max_i_ka=1.0,
                    name=f"Feeder_Line_{i}"
                )
                
                # 3. Transformer (HV -> Dist MV)
                sn_mva, vkr, vk = calculate_trafo_params(trans_kv)
                pp.create_transformer_from_parameters(
                    net, hv_bus=sub_hv_bus, lv_bus=dist_bus_idx,
                    sn_mva=sn_mva, vn_hv_kv=trans_kv, vn_lv_kv=dist_kv,
                    vkr_percent=vkr, vk_percent=vk, pfe_kw=0, i0_percent=0,
                    name=f"Main_Sub_Trafo_{i}"
                )
                
            except Exception as e:
                print(f"Skipping connection for dist bus {dist_bus_idx}: {e}")

        # Add ONE System Slack on the Transmission Side
        # Find the max voltage bus to place the slack
        if len(osm_to_pp_bus) > 0:
            max_v_bus = net.bus.loc[osm_to_pp_bus.values()]['vn_kv'].idxmax()
            pp.create_ext_grid(net, bus=max_v_bus, name="System Transmission Slack")

    # D. Connect Generators
    old_ext_grids = net.ext_grid.index
    net.ext_grid.drop(old_ext_grids, inplace=True)
    if generators_data and trans_coords_list:
        trans_tree = cKDTree(trans_coords_list)
        
        for i, gen in enumerate(generators_data):
            gen_lat, gen_lon = gen['pos']
            if(gen['source']!='solar'):
                # 1. Create Gen Bus (LV)
                gen_bus_lv = pp.create_bus(net, vn_kv=13.8, name=f"Gen_Bus_LV_{i}", geodata=(gen_lat, gen_lon))
                net.bus.at[gen_bus_lv, 'geo'] = json.dumps({"type": "Point", "coordinates": [gen_lon, gen_lat]})
                
                #pp.create_sgen(net, bus=gen_bus_lv, p_mw=10, q_mvar=0, name=f"Gen_{gen['source']}")
                pp.create_ext_grid(net, bus=gen_bus_lv, name=f"Gen_{gen['source']}")
                # 2. Find Nearest Grid Point
                _, idx = trans_tree.query((gen_lat, gen_lon))
                grid_bus = osm_to_pp_bus[trans_node_ids[idx]]
                grid_kv = net.bus.at[grid_bus, 'vn_kv']
                
                # 3. Create Step-up Station (HV Bus + Trafo)
                gen_bus_hv = pp.create_bus(net, vn_kv=grid_kv, name=f"Gen_Bus_HV_{i}", geodata=(gen_lat, gen_lon))
                net.bus.at[gen_bus_hv, 'geo'] = json.dumps({"type": "Point", "coordinates": [gen_lon, gen_lat]})
                
                sn, vkr, vk = calculate_trafo_params(grid_kv)
                pp.create_transformer_from_parameters(
                    net, hv_bus=gen_bus_hv, lv_bus=gen_bus_lv,
                    sn_mva=sn, vn_hv_kv=grid_kv, vn_lv_kv=13.8,
                    vkr_percent=vkr, vk_percent=vk, pfe_kw=0, i0_percent=0, name=f"Gen_Trafo_{i}"
                )
                
                # 4. Connect to Grid
                dist_km = ox.distance.great_circle(gen_lat, gen_lon, 
                                                trans_coords_list[idx][0], trans_coords_list[idx][1]) / 1000.0
                r_g, x_g = calculate_line_params(grid_kv)
                pp.create_line_from_parameters(
                    net, from_bus=gen_bus_lv, to_bus=grid_bus, length_km=max(0.1, dist_km),
                    r_ohm_per_km=r_g, x_ohm_per_km=x_g, c_nf_per_km=9, max_i_ka=1.0, name=f"Gen_Line_{i}"
                )

    # ======================================================
    # 5. STITCHING THE LAYERS (GRAPH)
    # ======================================================
    
    G_combined = G_trans.copy()
    nx.set_node_attributes(G_combined, 'transmission', 'type')
    
    # Create KDTree for graph stitching
    if len(G_combined.nodes) > 0:
        trans_nodes_list = list(G_combined.nodes)
        trans_coords = np.array([[G_combined.nodes[n]['y'], G_combined.nodes[n]['x']] 
                                 for n in trans_nodes_list])
        trans_tree_graph = cKDTree(trans_coords)
    else:
        trans_tree_graph = None
        
    # A. Add Connections for EACH Distribution Substation
    if trans_tree_graph:
        for i, dist_bus_idx in enumerate(dist_sub_indices):
            try:
                coords = json.loads(net.bus.at[dist_bus_idx, 'geo'])['coordinates']
                sub_lon, sub_lat = coords[0], coords[1]
                sub_id = f"dist_sub_{dist_bus_idx}"
                
                G_combined.add_node(sub_id, y=sub_lat, x=sub_lon, type='substation')
                
                _, idx = trans_tree_graph.query((sub_lat, sub_lon))
                closest_node = trans_nodes_list[idx]
                G_combined.add_edge(sub_id, closest_node, type='feeder_substation', weight=1)
            except: pass

    # B. Add Generators to Graph
    for gen in generators_data:
        G_combined.add_node(gen['id'], y=gen['pos'][0], x=gen['pos'][1], 
                            type='generator', source=gen['source'])
        
        if trans_tree_graph:
            _, idx = trans_tree_graph.query(gen['pos'])
            closest_node = trans_nodes_list[idx]
            G_combined.add_edge(gen['id'], closest_node, type='gen_connection', weight=1)

    print("--- Integration Complete ---")
    return net, G_combined

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology
from BayesGrid.src.bayesgrid.grid_generator import (
    BayesianPowerModel, 
    BayesianDurationModel, 
    BayesianFrequencyModel, 
    BayesianImpedanceModel
)
from BayesGrid.src.bayesgrid.grid_generator import (
    save_power_phase_samples, 
    save_bus_metric_samples, 
    save_impedance_samples
)

def generate_and_save_full_system_data(global_net, n_samples=1000, output_prefix="full_system"):
    """
    Takes the FULL global network (after voltage levels applied), splits it into 
    feeders dynamically, runs Bayesian models, and saves the consolidated results 
    mapped to the original global indices.
    """
    print(f"--- Starting Full System Generation (Splitting Global Net of {len(global_net.bus)} buses) ---")
    
    # =========================================================
    # 1. PRESERVE GLOBAL INDICES
    # =========================================================
    # We explicitly save the current global index to a column.
    # select_subnet will carry this column to the subnets.
    global_net.bus['global_index'] = global_net.bus.index
    global_net.line['global_index'] = global_net.line.index
    
    # Initialize Accumulators
    acc_power, acc_phases = [], []
    acc_freq, acc_dur = [], []
    acc_r, acc_x = [], []
    
    acc_bus_ids = []  # Will hold the global indices found in each feeder
    acc_line_ids = []

    # Initialize Models
    bpm = BayesianPowerModel(total_demand=None) 
    bdm = BayesianDurationModel()
    bfm = BayesianFrequencyModel()
    bim = BayesianImpedanceModel()

    # =========================================================
    # 2. SPLIT & PROCESS LOOP
    # =========================================================
    # We iterate over every source (External Grid) in the global system
    # This automatically handles the forest topology
    
    # Create a graph once for faster topology search
    nx_graph = pp.topology.create_nxgraph(global_net)
    
    processed_count = 0
    total_feeders = len(global_net.ext_grid)

    for ext_grid_idx in global_net.ext_grid.index:
        processed_count += 1
        print(f"\nProcessing Feeder {processed_count}/{total_feeders} (Source ID: {ext_grid_idx})...")
        
        try:
            # A. Identify Feeder Components
            root_bus = global_net.ext_grid.at[ext_grid_idx, 'bus']
            
            # Get all buses electrically connected to this source
            feeder_buses = list(pp.topology.connected_component(nx_graph, root_bus))
            
            # B. Extract Subnet (Independent Feeder)
            # This keeps the 'global_index' column we created earlier
            feeder_net = pp.select_subnet(global_net, buses=feeder_buses, include_results=False)
            
            # C. Capture IDs & Reset Index
            # These are the ORIGINAL GLOBAL INDICES required for the final CSV
            # current_bus_ids = feeder_net.bus['global_index'].values
            # current_line_ids = feeder_net.line['global_index'].values
            
            # Reset indices to 0..N for the Bayesian Matrix math to work
            # (using your optimized reset logic or pp.create_continuous_bus_index)
            pp.create_continuous_bus_index(feeder_net, start=0)
            # Ensure elements are reset too (lines, trafos)
            for elm in ['line', 'trafo', 'ext_grid', 'load', 'sgen']:
                if not feeder_net[elm].empty:
                    feeder_net[elm].reset_index(drop=True, inplace=True)

            current_bus_ids = feeder_net.bus['global_index'].values
            current_line_ids = feeder_net.line['global_index'].values
    

            # D. Run Bayesian Models
            ph, p = bpm.generate_data(net=feeder_net)
            dur = bdm.generate_data(net=feeder_net)
            freq = bfm.generate_data(net=feeder_net)
            r, x = bim.generate_data(net=feeder_net)

            # E. Slice Samples
            p = p[:n_samples, :, :]
            ph = ph[:n_samples, :]
            dur = dur[:n_samples, :]
            freq = freq[:n_samples, :]
            r = r[:n_samples, :]
            x = x[:n_samples, :]

            # F. Accumulate
            acc_power.append(p)
            acc_phases.append(ph)
            acc_freq.append(freq)
            acc_dur.append(dur)
            acc_r.append(r)
            acc_x.append(x)
            
            # Store as Pandas Series/Index for easy concat later
            acc_bus_ids.append(pd.Series(current_bus_ids))
            acc_line_ids.append(pd.Series(current_line_ids))
            
        except Exception as e:
            print(f"Skipping feeder {ext_grid_idx} due to error: {e}")
            continue

    # =========================================================
    # 3. CONSOLIDATE & SAVE
    # =========================================================
    print("\nConsolidating data from all feeders...")
    
    # Stack arrays horizontally (axis=1 is buses/lines)
    full_power = np.concatenate(acc_power, axis=1) 
    full_phases = np.concatenate(acc_phases, axis=1)
    full_freq = np.concatenate(acc_freq, axis=1)
    full_dur = np.concatenate(acc_dur, axis=1)
    full_r = np.concatenate(acc_r, axis=1)
    full_x = np.concatenate(acc_x, axis=1)
    
    # Reconstruct the Global Index List
    full_bus_index = pd.concat(acc_bus_ids)
    full_line_index = pd.concat(acc_line_ids)
    
    # Verification
    print(f"Global Net Buses: {len(global_net.bus)} | Recovered Data Buses: {len(full_bus_index)}")
    print(f"Global Net Lines: {len(global_net.line)} | Recovered Data Lines: {len(full_line_index)}")

    print(f"Saving files with prefix: {output_prefix}...")

    # 1. Save Power and Phase
    save_power_phase_samples(
        gen_phases=full_phases,
        gen_power=full_power,
        bus_index=full_bus_index,  # This passes the original global indices
        phase_map=bpm.get_phase_map(),
        n_samples=n_samples,
        output_path=f'{output_prefix}_bus_power_and_phase_SAMPLES.csv'
    )

    # 2. Save Frequency
    save_bus_metric_samples(
        gen_data=full_freq,
        col_name='CAIFI_FIC',
        bus_index=full_bus_index,
        n_samples=n_samples,
        output_path=f'{output_prefix}_bus_frequency_SAMPLES.csv'
    )

    # 3. Save Duration
    save_bus_metric_samples(
        gen_data=full_dur,
        col_name='CAIDI_DIC',
        bus_index=full_bus_index,
        n_samples=n_samples,
        output_path=f'{output_prefix}_bus_duration_SAMPLES.csv'
    )

    # 4. Save Impedance
    save_impedance_samples(
        gen_r=full_r,
        gen_x=full_x,
        line_index=full_line_index, # This passes the original global line indices
        n_samples=n_samples,
        output_path=f'{output_prefix}_line_impedance_SAMPLES.csv'
    )

    print("--- All Data Saved ---")
