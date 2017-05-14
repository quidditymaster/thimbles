from copy import copy
from . import Parameter, Model
import networkx as nx


def parent_edges(node):
    if isinstance(node, Parameter):
        mapped_mods = node.mapped_models
        n_mapped = len(mapped_mods)
        if n_mapped == 0:
            return []
        elif n_mapped == 1:
            return [(node.mapped_models[0], node)] #this is assuming each parameter may have only a single parent model
        else:
            raise ValueError("multiple models map to this parameter")
    elif isinstance(node, Model):
        return [(param, node) for param in node.parameters]


def child_edges(node):
    if isinstance(node, Parameter):
        return [(node, mod) for mod in node.models.contexts]
    elif isinstance(node, Model):
        return [(node, node.output_p)]


def extract_influence_graph(nodes, direction="upstream"):
    if direction=="upstream":
        edge_maker = parent_edges
    elif direction == "downstream":
        edge_maker = child_edges
    else:
        raise ValueError("direction choice {} not understood".format(direction))
    #make the graph
    influence_graph = nx.DiGraph()
    #seed the node stack
    if isinstance(nodes, (Parameter, Model)):
        nodes = [nodes]
    front_nodes = copy(nodes)
    #keep track of traversed nodes
    visited_nodes = set()
    while len(front_nodes) > 0:
        cnode = front_nodes.pop()
        if not cnode in visited_nodes:
            visited_nodes.add(cnode)            
            edges = edge_maker(cnode)
            for edge in edges:
                influence_graph.add_edge(*edge)
                front_nodes.extend(edge)
    return influence_graph


def execute_path(path, root_value, override, copy_override=True):
    if override is None:
        override = {}
    assert isinstance(path[0], Parameter)
    assert all([isinstance(m, Model) for m in path[1::2]])
    last_output = root_value
    if copy_override:
        rep_d = copy(override)
    else:
        rep_d = override
    for p_idx in range(int(len(path)//2)):
        param = path[2*p_idx]
        model = path[2*p_idx+1]
        rep_d[param] = last_output
        last_output = model(override=rep_d)
    return last_output


def find_all_paths(source, target, influence_graph=None):
    if influence_graph is None:
        influence_graph = extract_influence_graph(target, direction="upstream")
    return nx.algorithms.simple_paths.all_simple_paths(influence_graph, source, target)

def model_type_labeler(model):
    return model.model_class

def parameter_type_labeler(param):
    return param.parameter_class

def downstream_parameter_labeler(param):
    dsmods = param.models
    if len(param.models._aliases) > 0:
        return param.models._aliases[0].name
    else:
        return parameter_type_labeler(param)

def influence_graph_gv(
        influence_graph,
        model_labeler=None,
        parameter_labeler=None,
        graph_name="influence",
):
    if model_labeler is None:
        model_labeler = model_type_labeler
    if parameter_labeler is None:
        parameter_labeler = downstream_parameter_labeler
    
    gv = ["digraph {}".format(graph_name) +  " {"]
    nodes = influence_graph.nodes()
    node_ids = {nodes[i]:"{}".format(i) for i in range(len(nodes))}
    node_labels = {}
    nodestr = '{node_id} [shape={shape} label="{label}"];'
    for node in influence_graph.nodes():
        node_id = node_ids[node]
        if isinstance(node, Parameter):
            shape = "oval"
            label = parameter_labeler(node)
        elif isinstance(node, Model):
            shape = "box"
            label = model_labeler(node)
        
        gv.append(nodestr.format(shape=shape, node_id=node_id, label=label))
    ###
    for n1, n2 in influence_graph.edges():
        n1id = node_ids[n1]
        n2id = node_ids[n2]
        gv.append("{} -> {};".format(n1id, n2id))
    gv.append("}")
    return "\n".join(gv)

