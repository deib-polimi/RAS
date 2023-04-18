import networkx as nx
from collections import defaultdict

from matplotlib import pyplot as plt
from networkx import topological_sort

from applications import Application1
from applications import Application2
from Task import Task


def DFSUtil(node, graph, visited, path, paths):
    visited[node] = True
    path.append(node)
    if not graph[node]:
        paths.append(path.copy())
    for neighbor in graph[node]:
        if not visited[neighbor]:
            DFSUtil(neighbor, graph, visited, path, paths)
    path.pop()
    visited[node] = False


def getAllFullPaths(graph, start):
    visited = defaultdict(bool)
    paths = []
    DFSUtil(start, graph, visited, [], paths)
    return paths

# calculate the response time for a given single function (app) given the number of call and a specific application
# all details are encapsulated in the function
def comput_simple_RT(function_req_node, app):
    rt = app.__computeRT__(function_req_node)
    return rt


def build_rt_dag(dag, start_node):  # OK
    new_dag = nx.DiGraph()
    nx.set_node_attributes(new_dag, new_dag.nodes, 'node_rt')
    nx.set_edge_attributes(new_dag, new_dag.edges, 'req')
    paths = getAllFullPaths(dag, start_node)
    list = []
    for path in paths:
        node1 = path[0]
        path.pop(0)
        if node1 not in list:
            req = dag.nodes[node1]['node_req']
            app = dag.nodes[node1]['rt_model']  # convert string to instance
            rt = round(comput_simple_RT(req, app), 5)
            new_dag.add_node(node1)
            new_dag.nodes[node1]['node_rt'] = rt
            list.append(node1)
        for node2 in path:
            if node2 not in list:
                new_dag.add_node(node2)  # connect node 1 to 2 (A-->B)
                req = dag.nodes[node2]['node_req']
                app = dag.nodes[node2]['rt_model']  # convert string to instance
                rt = round(comput_simple_RT(req, app), 5)
                new_dag.add_node(node2)
                new_dag.nodes[node2]['node_rt'] = rt
                list.append(node2)
            weight = dag.edges[node1, node2]["req"]
            new_dag.add_edge(node1, node2, req=weight)
            node1 = node2
    return new_dag


# return the path of nodes with the highest total response time
# variable dag must be felled by Node with RT
def max_total_req_path(dag, start_node):  # OK
    paths = getAllFullPaths(dag, start_node)
    selected_path = []
    total_rt_of_selected_path = 0
    for path in paths:
        new_total_rt = 0
        for node in path:
            new_total_rt += dag.nodes[node]['node_rt']
        if new_total_rt > total_rt_of_selected_path:
            total_rt_of_selected_path = new_total_rt
            selected_path = path
    return selected_path


# DAG of RT functions ( only for the path with the highest total response time)
#  input DAG-with RT in each node and req for edges and the heaviest PATH
def build_dag_max_path_rt_dag(dag, path):
    new_dag = nx.DiGraph()
    nx.set_node_attributes(new_dag, new_dag.nodes, 'node_rt')
    nx.set_edge_attributes(new_dag, new_dag.edges, 'req')
    node1 = ''
    path_copy = path.copy()
    if path_copy is not None:
        node1 = path_copy[0]
        new_dag.add_node(node1)
        new_dag.nodes[node1]['node_rt'] = dag.nodes[node1]['node_rt']
        path_copy.pop(0)
    for node2 in path_copy:
        new_dag.add_node(node2)
        new_dag.add_edge(node1, node2)
        new_dag.nodes[node2]['node_rt'] = dag.nodes[node2]['node_rt']
        weight = dag.edges[node1, node2]["req"]
        new_dag.add_edge(node1, node2, req=weight)
        node1 = node2
    return new_dag


# Compute the RT of the selected path
def comput_max_RT(rt_dag, selected_path):
    total_rt = 0
    for node in selected_path:
        total_rt += rt_dag.nodes[node]['node_rt']
    return total_rt


def print_dag(dag, node_content):
    pos = nx.kamada_kawai_layout(dag)

    # Set figure size and margins
    fig, ax = plt.subplots(figsize=(9, 12))
    fig.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.05)

    # Draw nodes with labels and attribute values
    node_labels = {node: f"{node} ({attrs[node_content]})" for node, attrs in dag.nodes(data=True)}
    nx.draw_networkx_nodes(dag, pos, node_color='none', edgecolors='blue', node_size=2500)
    nx.draw_networkx_labels(dag, pos, labels=node_labels, font_size=12, font_weight='700', font_color='blue')

    # Draw edges with labels and attribute values
    edge_labels = nx.get_edge_attributes(dag, 'req')
    nx.draw_networkx_edges(dag, pos, width=2, alpha=1, edge_color='blue')
    nx.draw_networkx_edge_labels(dag, pos, edge_labels=edge_labels, font_size=12, font_color='blue')
    ax.axis('off')
    plt.show()


# New method
# Return a list of nodes in order of given attribute
def list_nodes_by_attribute(dag, attribute):
    return sorted(dag.nodes, key=lambda n: dag.nodes[n][attribute])

if __name__ == "__main__":
    dg = nx.DiGraph([("A", "B", {"req": 10}), ("B", "C", {"req": 30}),
                     ("C", "D", {"req": 30}), ("D", "E", {"req": 40}),
                     ("E", "F", {"req": 80}), ("A", "G", {"req": 10}),
                     ("A", "H", {"req": 10}), ("H", "E", {"req": 40}),
                     ("A", "I", {"req": 10}), ("I", "D", {"req": 10})
                     ])
    # nx.set_node_attributes(dg, dg.nodes, 'total_node_req')
    nx.set_node_attributes(dg, dg.nodes, 'rt_model')
    # nx.set_edge_attributes(dg, dg.nodes, 'req')
    dg.nodes['A']['node_req'] = 10
    dg.nodes['B']['node_req'] = 30
    dg.nodes['C']['node_req'] = 30
    dg.nodes['D']['node_req'] = 40
    dg.nodes['E']['node_req'] = 80
    dg.nodes['F']['node_req'] = 90
    dg.nodes['G']['node_req'] = 10
    dg.nodes['H']['node_req'] = 40
    dg.nodes['I']['node_req'] = 10

    dg.nodes['A']['priority'] = [1]
    dg.nodes['B']['priority'] = [2]
    dg.nodes['C']['priority'] = [3]
    dg.nodes['D']['priority'] = [4, 11]
    dg.nodes['E']['priority'] = [5, 9]
    dg.nodes['F']['priority'] = [6]
    dg.nodes['G']['priority'] = [7]
    dg.nodes['H']['priority'] = [8]
    dg.nodes['I']['priority'] = [10]

    dg.nodes['A']['rt_model'] = Application1(sla=0.1)
    dg.nodes['B']['rt_model'] = Application1(sla=0.1)
    dg.nodes['C']['rt_model'] = Application1(sla=0.1)
    dg.nodes['D']['rt_model'] = Application1(sla=0.1)
    dg.nodes['E']['rt_model'] = Application1(sla=0.1)
    dg.nodes['F']['rt_model'] = Application1(sla=0.1)
    dg.nodes['G']['rt_model'] = Application2(sla=0.1)
    dg.nodes['H']['rt_model'] = Application2(sla=0.1)
    dg.nodes['I']['rt_model'] = Application1(sla=0.1)


    def sub_dag(self, startNodeName):
        subdag = {}
        visited = {startNodeName}
        queue = [startNodeName]

        while queue:
            node = queue.pop(0)
            for neighbor in dg[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    subdag.setdefault(node, set()).add(neighbor)
        return subdag

    # Print DAG with total of calls on the nodes
    print_dag(dag=dg, node_content='node_req')  # OK

    # Print the DAG with RT
    rt_dg = build_rt_dag(dg, 'A')
    print_dag(dag=rt_dg, node_content='node_rt')

    # Print the heaviest RT path
    max_path = max_total_req_path(rt_dg, 'A')
    rt_dg1 = build_dag_max_path_rt_dag(rt_dg, max_path)
    print_dag(dag=rt_dg1, node_content='node_rt')

   # sorted_nodes = list_nodes_by_attribute(dg, 'priority'[0])
    #for node in sorted_nodes:
    #print(sorted_nodes)
