import copy
from collections import defaultdict

import networkx as nx
from matplotlib import pyplot as plt


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



class DAG:
    def __init__(self, dag, startNodename):
        self.dag = dag
        self.startNodeName=startNodename

    # Join multiple lists of node names (Strings) into one without repetitions
    def uniqueList(self, lists):
        listNames = []
        for list in lists:
            for i in range(0, len(list)):
                nodeName = list[i]
                if nodeName not in listNames:
                    listNames.append(nodeName)
        return listNames

    def getNode(self, nodeName):
        return self.dag.nodes[nodeName]['node']

    def sub_dag(self, startNodeName):
        subdag = {}
        visited = {startNodeName}
        queue = [startNodeName]

        while queue:
            node = queue.pop(0)
            for neighbor in self.dag[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    subdag.setdefault(node, set()).add(neighbor)
        return subdag

    # convert a Dag into a list given a start point -- DONE
    def toList(self, startNode):
        lists = getAllFullPaths(self.dag, startNode)
        newList = []
        listNames = []
        for nodeNamelist in lists:
            nodeName = nodeNamelist[0]
            node =self.dag.nodes[nodeName]['node']
            nodeNamelist.pop(0)
            if nodeName not in listNames:
                newList.append(node)
                listNames.append(nodeName)
            for nodeName in nodeNamelist:
                if nodeName not in listNames:
                    node = self.dag.nodes[nodeName]['node']
                    newList.append(node)
                    listNames.append(nodeName)
        return newList

    # sum and return the RT of given Node (given by name) considering dependencies
    def getNodeRT(self, nodeName):
        rt = self.getNode(nodeName).total_rt
        if rt == 0:
            rt = self.getNode(nodeName).app.RT
        return rt

    # update number of users for all apps from pos toward
    def updateUsersList(self, children, newusers, t):
        if len(children)==0:
            return

        for child in children:
            node = self.dag.nodes[child]['node']
            app, gen = node.app, node.generator
            # currentusersapp=app.numberusers # first is zero
            childNewusers = int(gen.tick(t))
            # we consider the last users generated in father and local child new users
            total_new_users = app.users+childNewusers+newusers
            app.users = total_new_users
            new_rt=node.app.getRT(total_new_users)
            app.RT=new_rt
            node.total_rt = new_rt  # Set Local RT
            # print(child, '.Users=', app.users)
        for child in children:
            node = self.dag.nodes[child]['node']
            childrenList = self.get_children(child)
            self.updateUsersList(childrenList, node.app.users, t)

    def resetUsers(self, start):
        dagList = self.toList(start)
        for node in dagList:
            self.dag.nodes[node.name]['node'].app.users=0

    # Only simple DAG was considered and set Local (simple) RT
    def updateUsersDAG(self, startNodeName, t): # start is
        node = self.dag.nodes[startNodeName]['node']
        app, gen = node.app, node.generator
        # currentusersapp=app.numberusers # first is zero
        newusers = int(gen.tick(t))
        users_app = app.users + newusers
        app.users = users_app
        app.setRT(users_app)  # Set Local RT
        node.total_rt = app.RT
        childrenList = self.get_children(startNodeName)
        self.updateUsersList(childrenList, newusers,t)

    def setCores(self, startNodeName, t):
        listDAG=self.toList(startNodeName)
        for node in listDAG:
            app = node.app
            total_rt = self.getNodeRT(node.name)
            mo = node.monitoring
            cont = node.controller
            mo.tick(t, app.RT, total_rt, app.users, app.cores) # TODO: add local RT
            cores_app = cont.tick(t)
            app.cores = cores_app

    def getControllers(self):
        controllers=[]
        listNodes=self.toList(self.startNodeName)
        for node in listNodes:
            controllers.append(node.controller)
        return controllers

    def getGenerators(self):
        generators  = []
        listNodes = self.toList(self.startNodeName)
        for node in listNodes:
            generators .append(node.generator)
        return generators

    def getMonitorings(self):
        monitorings = []
        listNodes = self.toList(self.startNodeName)
        for node in listNodes:
            monitorings.append(node.monitoring)
        return monitorings

    def get_children(self, node):
        children = []
        edges = self.dag.out_edges(node, data=True)
        for edge in edges:
            children.append(edge[1])
        return children

    def get_nodes_with_children_no_grandChildren(self, dag):
        nodes_with_no_children_children = set()
        for node in dag.nodes():
            if dag.out_degree(node) == 0:
                continue
            all_children_have_no_children = True
            for child in dag.successors(node):
                if dag.out_degree(child) > 0:
                    all_children_have_no_children = False
                    break
            if all_children_have_no_children:
                nodes_with_no_children_children.add(node)
        return nodes_with_no_children_children

    # Calculate the total RT for each Node given the local one
    def getTotalNodeRT(self,  rootNodeName):

        #list=self.get_children(rootNodeName)
        totalRT=0
        #for nodeName in list:
        exec_order = self.get_unique_edge_values(rootNodeName)
        # print('SYNC=', exec_order)
        for edge_value in exec_order:
            max_rt_node = self.get_max_rt_child_node(self.dag, rootNodeName, edge_value)
            totalRT += max_rt_node.total_rt  # sum the RT of children, use only the max for async nodes
        totalRT += self.getNodeRT(rootNodeName)
        return totalRT

    def setAllRT(self):
        cloned_dag = copy.deepcopy(self.dag)
        visited_nodes = []
        while len(cloned_dag) > 1:
            nodes_names_without_grand_child = self.get_nodes_with_children_no_grandChildren(cloned_dag)
            for nodeName in nodes_names_without_grand_child:
                new_rt = self.getTotalNodeRT(nodeName)
                # print('RT-Local=', self.getNode(nodeName).app.RT)
                # print('RT=', new_rt)
                self.getNode(nodeName).total_rt = new_rt  # set RT to our MAP
                cloned_dag.nodes[nodeName]['node'].total_rt = new_rt  # set RT to the cloned MAP
                children = self.get_children(nodeName)
                visited_nodes.append(children)
            unique_list = self.uniqueList(visited_nodes)
            cloned_dag.remove_nodes_from(unique_list)

    def get_children_nodes(self, start_node, edge_value):
        children_nodes = []
        # Check all outgoing edges of the start node
        for child, edge in self.dag[start_node].items():
            # print(edge)
            if edge == edge_value:
                children_nodes.append(child)
        return children_nodes

    def get_max_rt_child_node(self, dag, start_nodeName, edge_value):
        children_nodes_names = self.get_children_nodes(start_nodeName, edge_value)
        max_rt_child_node_name = children_nodes_names[0]
        max_nod = self.getNode(max_rt_child_node_name)
        # Check all outgoing edges of the start node
        for child_name, edge in dag[start_nodeName].items():
            # print(edge)
            if edge == edge_value:
                child = self.getNode(child_name)
                if child.total_rt > max_nod.total_rt:
                    max_nod = child
        return max_nod

    def get_unique_edge_values(self, nodeName):
        edge_values = []

        # Check all outgoing edges of the node
        for _, edge in self.dag[nodeName].items():
            if edge not in edge_values:
                edge_values.append(edge)

        return edge_values
                # VISUALIZATION

    # For MAP visualization only
    def updateDAGUsersRTForVisualization(self, start):  # start is the root of the MAP
        nodeNameList=self.uniqueList(getAllFullPaths(self.dag, start))
        for nodeName in nodeNameList:
            self.dag.nodes[nodeName]['users'] = self.dag.nodes[nodeName]['node'].app.users
            self.dag.nodes[nodeName]['rt'] = round(self.dag.nodes[nodeName]['node'].total_rt, 6)


    def print_dag(self, node_content):
        pos = nx.spring_layout(self.dag)
            #nx.kamada_kawai_layout(self.dag)
        # Set figure size and margins
        fig, ax = plt.subplots(figsize=(9, 12))
        fig.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.05)

        # Draw nodes with labels and attribute values
        node_labels = {node: f"{node} ({attrs[node_content]})" for node, attrs in self.dag.nodes(data=True)}
        nx.draw_networkx_nodes(self.dag, pos, node_color='none', edgecolors='blue', node_size=2500)
        nx.draw_networkx_labels(self.dag, pos, labels=node_labels, font_size=12, font_weight='700', font_color='blue')

        # Draw edges with labels and attribute values
        edge_labels = nx.get_edge_attributes(self.dag, 'sync')
        nx.draw_networkx_edges(self.dag, pos, width=2, alpha=1, edge_color='blue')
        nx.draw_networkx_edge_labels(self.dag, pos, edge_labels=edge_labels, font_size=12, font_color='blue')
        ax.axis('off')
        plt.show()

