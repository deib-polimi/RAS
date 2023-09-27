import copy
import csv
from collections import defaultdict
import networkx as nx
from scipy.io import savemat
import os
import pygraphviz
import numpy as np
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


def to_mil(number):
    return number*1000

def inv_mil(number):
    return number/1000


class ComplexDAG:
    def __init__(self, dag, startNodename):
        self.dag = dag
        self.startNodeName = startNodename
        nx.set_node_attributes(dag, dag.nodes, 'users')
        nx.set_node_attributes(dag, dag.nodes, 'node')
        nx.set_edge_attributes(dag, dag.edges, 'sync')  # synchrone or asynchrone calls
        nx.set_edge_attributes(dag, dag.edges, 'times')  # for cycles
        nx.set_node_attributes(dag, dag.nodes, 'app')

        nx.set_node_attributes(dag, dag.nodes, 'total_node_req')
        nx.set_node_attributes(dag, dag.nodes, 'rt')
        nx.set_node_attributes(dag, dag.nodes, 'st')
        nx.set_node_attributes(dag, dag.nodes, 'lrt')
        nx.set_node_attributes(dag, dag.nodes, 'cores')
        nx.set_node_attributes(dag, dag.nodes, 'cores_deviation')
        nx.set_node_attributes(dag, dag.nodes, 'rt_deviation')
        nx.set_node_attributes(dag, dag.nodes, 'lrt_deviation')

        self.maxTotalRTs = {}
        self.minTotalRTs = {}
        self.AvgTotalRTs = {}
        self.totalRts = {}
        self.deviationTotalRTs = {}

        self.maxCores = {}
        self.minCores = {}
        self.avgCores = {}
        self.cores = {}
        self.deviationCores = {}
        self.maxLocalRTs = {}
        self.minLocalRTs = {}
        self.AvgLocalRTs = {}
        self.localRts = {}
        self.deviationLocalRTs = {}

        self.slaViolations = {}
        self.slaNoViolations = {}
        self.avgSlaViolations = {}
        self.deviationSlaViolations = {}

        self.maxMinNotIntialized = True
        self.contInterestingRTsCores = 0

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
        # rt = self.getNode(nodeName).total_rt
        # if rt == 0:
        rt = self.getNode(nodeName).app.RT
        return rt

    def getNodeWeight(self, nodeName):
        weight = self.getNode(nodeName).subtotal_weight
        if weight == 0:
            weight = self.getNode(nodeName).app.weight
        return weight

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
        # self.updateUsersList(childrenList, newusers, t)
        self.updateUsersListForCycles(startNodeName,  childrenList , t)

    # called after users updated, only for cycles
    def updateUsersListForCycles(self, father, children,t):
        if len(children) == 0:
            return
        for child in children:
            childNode = self.dag.nodes[child]['node']
            app, gen = childNode.app, childNode.generator
            childNewusers = int(gen.tick(t))
            fatherUsers=self.getNode(father).app.users
            times = self.getEdgeValue(father, childNode.name, 'times')
            # we consider the last users generated in father and local child new users
            ti = 1 if (times < 1 or None) else times
            fathers=childNode.fathers
            previousFatherUsers = 0

            if father in fathers:
                    previousFatherUsers = fathers[father]
                    fathers[father] = fatherUsers
            else:
                    fathers.update({father: fatherUsers})

            # remove the previous added requests from father and add the updated ones
            total_new_users = app.users + fatherUsers * ti + childNewusers-previousFatherUsers
            app.users = total_new_users
            new_rt = childNode.app.getRT(total_new_users)
            app.RT = new_rt  # Set Local RT
            childNode.total_rt = app.RT
            childrenList = self.get_children(child)
            self.updateUsersListForCycles(child, childrenList, t)

    def getEdgeValue(self, node1, node2, attribute):
        return self.dag.edges[node1, node2][attribute]

    def resetUsers(self, start):
        dagList = self.toList(start)
        for node in dagList:
            self.dag.nodes[node.name]['node'].app.users = 0
            self.dag.nodes[node.name]['node'].fathers = {}

    def resetweights(self, start):
        dagList = self.toList(start)
        for node in dagList:
            self.dag.nodes[node.name]['node'].subtotal_weight=self.dag.nodes[node.name]['node'].app.weight
    # Only simple DAG was considered and set Local (simple) RT


    def setCores(self, startNodeName, t):
        listDAG=self.toList(startNodeName)
        for node in listDAG:
            app = node.app
            total_rt = self.getTotalNodeRT(node.name)
            mo = node.monitoring
            cont = node.controller
            mo.tick(t, app.RT, total_rt, app.users, app.cores)  # TODO: add local RT
            cores_app = cont.tick(t)
            app.cores = cores_app
            # print('T-', t, '/RT', app.RT, '/TRT', total_rt, '/U', app.users, '/C', app.cores)

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

    def getTotalNodeRT(self,  rootNodeName):
        totalRT=0
        #for nodeName in list:
        exec_order = self.get_unique_edge_values(rootNodeName)
        for edge_value in exec_order:
            max_rt_node = self.get_max_rt_child_node(self.dag, rootNodeName, edge_value)
            totalRT += max_rt_node.total_rt  # sum the RT of children, use only the max for async nodes
        totalRT += self.getNodeRT(rootNodeName)
        return totalRT

    def getTotalNodeWeight(self,  rootNodeName):
        #list=self.get_children(rootNodeName)
        totalWeight=0
        #for nodeName in list:
        exec_order = self.get_unique_edge_values(rootNodeName)
        for edge_value in exec_order:
            max_rt_node = self.get_max_rt_child_node(self.dag, rootNodeName, edge_value)
            totalWeight += max_rt_node.subtotal_weight  # sum the Weight of children, use only the max for parallel nodes
        totalWeight += self.getNodeWeight(rootNodeName)
        return totalWeight

    def setAllRT(self):
        cloned_dag = copy.deepcopy(self.dag)
        visited_nodes = []
        while len(cloned_dag) > 1:
            nodes_names_without_grand_child = self.get_nodes_with_children_no_grandChildren(cloned_dag)
            # for nodea in nodes_names_without_grand_child:
            #     print(nodea)
            for nodeName in nodes_names_without_grand_child:
                new_rt = self.getTotalNodeRT(nodeName)
                self.getNode(nodeName).total_rt = new_rt  # set RT to our MAP
                cloned_dag.nodes[nodeName]['node'].total_rt = new_rt  # set RT to the cloned MAP
                children = self.get_children(nodeName)
                visited_nodes.append(children)
            unique_list = self.uniqueList(visited_nodes)
            cloned_dag.remove_nodes_from(unique_list)


    def get_nodes_with_children_no_grandChildren(self, dag):
        nodes_with_no_grand_children = set()
        for node in dag.nodes():
            if dag.out_degree(node) == 0:
                continue
            all_children_have_no_children = True
            for child in dag.successors(node):
                if dag.out_degree(child) > 0:
                    all_children_have_no_children = False
                    break
            if all_children_have_no_children:
                nodes_with_no_grand_children.add(node)
        return nodes_with_no_grand_children
    def setAllWeights(self):
        cloned_dag = copy.deepcopy(self.dag)
        visited_nodes = []
        while len(cloned_dag) > 1:
            nodes_names_without_grand_child = self.get_nodes_with_children_no_grandChildren(cloned_dag)
            for nodeName in nodes_names_without_grand_child:
                new_weight = self.getTotalNodeWeight(nodeName)
                self.getNode(nodeName).subtotal_weight = new_weight  # set Global RT (Total weight for a Node) to our DAG
                cloned_dag.nodes[nodeName]['node'].subtotal_weight = new_weight   # set Global RT (Total weight for a Node) to our cloned DAG
                children = self.get_children(nodeName)
                visited_nodes.append(children)
            unique_list = self.uniqueList(visited_nodes) # TODO consider just using children
            cloned_dag.remove_nodes_from(unique_list)

    def setST(self, alfa):
        # max = 0
        self.setAllWeights()
        for node in self.dag:
            new_node = self.dag.nodes[node]['node']
            # newst= new_node.app.sla*alfa*new_node.subtotal_weight/new_node.total_weight
            newst = alfa * new_node.app.weight / new_node.total_weight
            # print(new_node.name, "- weight[%.3f]- totalweight[%.3f]", (new_node.subtotal_weight, new_node.total_weight))
            # max = new_node.subtotal_weight if max <new_node.subtotal_weight else max
            new_node.controller.setST(newst)
        # print(max)

    # Given a sync value, return the list of nodes that are
    # linked with that sync value from a given start node
    def get_children_nodes(self, start_node, edge_value): # TODO (DONE)
        children_nodes = []
        # Check all outgoing edges of the start node
        for child, edge in self.dag[start_node].items():
            new_edge = {key: value for key, value in edge.items() if key == 'sync'}
            if new_edge['sync'] == edge_value:
                children_nodes.append(child)
        return children_nodes

    def get_max_rt_child_node(self, dag, start_nodeName, edge_value): # TODO (DONE)
        children_nodes_names = self.get_children_nodes(start_nodeName, edge_value)
        max_rt_child_node_name = children_nodes_names[0]
        max_nod = self.getNode(max_rt_child_node_name)
        # Check all outgoing edges of the start node
        for child_name, edge in dag[start_nodeName].items():
            dict_new_edge = {key: value for key, value in edge.items() if key == 'sync'}
            new_edge = dict_new_edge['sync']
            # child = self.getNode(child_name)
            if new_edge == edge_value:
                child = self.getNode(child_name)
                if child.total_rt > max_nod.total_rt:
                    max_nod = child
        return max_nod

    def get_unique_edge_values(self, nodeName): # TODO (DONE)
        aux_list = []
        edge_values = []
        for key, value in self.dag[nodeName].items():
            if 'sync' in value:
                # Add value to the array for key='sync'
                aux_list.append(value['sync'])
        # Check all outgoing edges of the node
        for sync_edge in aux_list:
            if sync_edge not in edge_values:
                edge_values.append(sync_edge)
        return edge_values

    # VISUALIZATION
    def __initializeRTs(self):
            for node in self.toList(self.startNodeName):
                rt = node.total_rt
                lrt = node.app.RT
                self.maxTotalRTs.update({node.name: rt})
                self.minTotalRTs.update({node.name: rt})
                self.AvgTotalRTs.update({node.name: rt})
                self.totalRts.update({node.name: [rt]})

                self.maxLocalRTs.update({node.name: lrt})
                self.minLocalRTs.update({node.name: lrt})
                self.AvgLocalRTs.update({node.name: lrt})
                self.localRts.update({node.name: [lrt]})
            self.contInterestingRTsCores = 1
    def __initializeCores(self):
        for node in self.toList(self.startNodeName):
            cores = node.app.cores
            self.maxCores.update({node.name: cores})
            self.minCores.update({node.name: cores})
            self.avgCores.update({node.name: cores})
            self.cores.update({node.name: [cores]})
            self.contInterestingRTsCores = 1

    def to_mil(self, dictionary):
        for key in dictionary.keys():
            dictionary.update({key: dictionary[key] * 1000})

    #Local
    def fillStatiscal(self, starpointUsers = 1):
        for node in self.toList(self.startNodeName):
            users = node.app.users
            if users < starpointUsers:
                break
            else:
                if self.maxMinNotIntialized:
                    self.__initializeCores()
                    self.__initializeRTs()
                    self.maxMinNotIntialized = False
                    return
                else:
                    nodeName = node.name
                    rt = node.total_rt
                    lrt = node.app.RT
                    cores = node.app.cores
                    total_violations = node.monitoring.getViolations()
                    total_Nviolations = node.monitoring.getNViolations()
                    self.totalRts[nodeName].append(rt)
                    self.localRts[nodeName].append(lrt)
                    self.cores[nodeName].append(cores)
                    self.slaViolations.update({nodeName: total_violations})
                    self.slaNoViolations.update({nodeName: total_Nviolations})
                    if self.maxTotalRTs[nodeName] < rt:
                        self.maxTotalRTs.update({nodeName: rt})

                    if self.minTotalRTs[nodeName] > rt:
                        self.minTotalRTs.update({nodeName: rt})

                    if self.maxLocalRTs[nodeName] < lrt:
                        self.maxLocalRTs.update({nodeName: lrt})

                    if self.minLocalRTs[nodeName] > lrt:
                        self.minLocalRTs.update({nodeName: lrt})

                    if self.maxCores[nodeName] < cores:
                        self.maxCores.update({nodeName: cores})

                    if self.minCores[nodeName] > cores:
                        self.minCores.update({nodeName: cores})
        self.contInterestingRTsCores += 1

    # Public
    def setStatistical(self):
        self.to_mil(self.minTotalRTs)
        self.to_mil(self.maxTotalRTs)
        self.to_mil(self.minLocalRTs)
        self.to_mil(self.maxLocalRTs)
        self.to_mil(self.minCores)
        self.to_mil(self.maxCores)

        for nodeName in self.totalRts.keys():
            node = self.getNode(nodeName)
            self.AvgTotalRTs.update({nodeName: round(to_mil(np.mean(self.totalRts[nodeName])), 1)})
            self.AvgLocalRTs.update({nodeName: round(to_mil(np.mean(self.localRts[nodeName])), 1)})
            self.avgCores.update({nodeName: round(to_mil(np.mean(self.cores[nodeName])), 1)})
            self.deviationTotalRTs.update({nodeName: round(to_mil(np.std(self.totalRts[nodeName])), 1)})

            self.deviationLocalRTs.update({nodeName: round(to_mil(np.std(self.localRts[nodeName])), 1)})
            self.deviationCores.update({nodeName: round(to_mil(np.std(self.cores[nodeName])), 1)})

            self.deviationSlaViolations.update({nodeName: node.monitoring.getTotalViolations()})

            self.avgSlaViolations.update({nodeName: node.monitoring.getTotalViolations()})
            # print(nodeName, node.local_sla, node.total_rt)
    # For DAG visualization only
    def updateForVisualization(self, start):  # start is the root of the MAP
        nodeNameList=self.uniqueList(getAllFullPaths(self.dag, start))
        for nodeName in nodeNameList:
            self.dag.nodes[nodeName]['users'] = self.dag.nodes[nodeName]['node'].app.users
            self.dag.nodes[nodeName]['st'] = round(to_mil(self.dag.nodes[nodeName]['node'].controller.st), 2)
            self.dag.nodes[nodeName]['rt'] = self.AvgTotalRTs[nodeName]     # get from dictionary
            self.dag.nodes[nodeName]['lrt'] = self.AvgLocalRTs[nodeName]  # get from dictionary
            self.dag.nodes[nodeName]['rt_deviation'] = self.deviationTotalRTs[nodeName]  # get from dictionary
            self.dag.nodes[nodeName]['lrt_deviation'] = self.deviationLocalRTs[nodeName]  # get from dictionary
            self.dag.nodes[nodeName]['cores'] = self.avgCores[nodeName]     # get from dictionary
            self.dag.nodes[nodeName]['cores_deviation'] = self.deviationCores[nodeName]  # get from dictionary
            self.dag.nodes[nodeName]['app'] = 'f'

    def print_dag(self, node_content=None, sync='sync', times='times', show_sync=True, show_times=True,
                  pos=None, seed_value=42, label=''):
            actualDirectCalledNodes=['f1']
            entrypoint_nodes = ['f1','f2', 'f11','f9', 'f14','f24']
            fontsize=8
            dagcolor='black'
            # seed = seed_value,
            if pos is None:
             pos = nx.spring_layout(self.dag, scale=2)

            #nx.kamada_kawai_layout(self.dag)
            # Set figure size and margins
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.subplots_adjust(left=0.001, right=1, top=1, bottom=0.01)
            node_labels = {node: node for node in self.dag.nodes}
            # Draw nodes with labels and attribute values
            if node_content is not None:
                node_labels = {node: f"{node} ({attrs[node_content]})" for node, attrs in self.dag.nodes(data=True)}
            nx.draw_networkx_labels(self.dag, pos, labels=node_labels, font_size=6, font_weight='10',
                                        font_color=dagcolor)
            nx.draw_networkx_nodes(self.dag, pos, node_color='none', edgecolors=dagcolor, node_size=2000)
            self.selectNode(entrypoint_nodes, pos, edgecolors=dagcolor, node_color='none', node_size=2000, linewidths=3)
            # Actual Direct called Nodes
            self.selectNode(actualDirectCalledNodes, pos, edgecolors=dagcolor, node_color='none', node_size=2000, linewidths=3)

            # Draw edges with labels and attribute values
            edge_labels_sync_decorated = nx.get_edge_attributes(self.dag, sync)
            edge_labels_times_decorated = nx.get_edge_attributes(self.dag, times)
            edge_labels_sync = nx.get_edge_attributes(self.dag, sync)  # 'sync'
            edge_labels = copy.deepcopy(edge_labels_sync)
            edge_labels_times = nx.get_edge_attributes(self.dag, times)

            for key in edge_labels_sync:
                edge_labels_sync_decorated[key] = f'({"s"}{edge_labels_sync[key]})'
                edge_labels_times_decorated[key] = f'({"t"}{edge_labels_times[key]})'
                edge_labels[key] = f'({"s"}{edge_labels_sync[key]},{"t"}{edge_labels_times[key]})'
            nx.draw_networkx_edges(self.dag, pos, width=1, alpha=1, edge_color=dagcolor, connectionstyle='arc3,rad=-0.02')
            if show_sync and show_times:
                nx.draw_networkx_edge_labels(self.dag, pos, edge_labels=edge_labels, font_size=fontsize, font_color=dagcolor)
            else:
                if show_sync:
                    nx.draw_networkx_edge_labels(self.dag, pos, edge_labels=edge_labels_sync_decorated, font_size=fontsize, font_color=dagcolor)
                else:
                    if show_times:
                        nx.draw_networkx_edge_labels(self.dag, pos, edge_labels=edge_labels_times_decorated, font_size=fontsize,
                                              font_color=dagcolor)
            # special_nodes = ["arf1", "arf9", "arf24", "arf11", "arf14"]
            # edges = [("arf1", "f1"), ("arf9", "f9"), ("arf24", "f24"), ("arf11", "f11"), ("arf14", "f14")]
            # newpos={
            #     # special arrows
            #     "arf1": (10, 10),
            #     "arf9": (6, 3),
            #     "arf24": (11, -1),
            #     "arf11": (13, 6),
            #     "arf14": (18, 4)
            # }
            # self.addNewNodes(special_nodes, edges)
            # self.drawNewNodes(nodeList=special_nodes, edgeList=edges,pos=newpos, edgecolors=dagcolor, node_color='lightblue', node_size=2000,
            #                 linewidths=3)

            ax = plt.gca()
            ax.margins(0.01)
            ax.axis('off')
            font1 = {'family': 'serif', 'color': dagcolor, 'size': 10}
            plt.title(node_content, fontdict=font1, loc='left')
            plt.title(seed_value, fontdict=font1, loc='left')

            if node_content is None:
                node_content = "simple"
            plt.savefig("complex-dag(%s)-%s.pdf" % (label, node_content))
            plt.show()
            plt.close()

    def selectNode(self, nodeList, pos=None, edgecolors='black', node_color='none', node_size=2000, linewidths=3):
        nx.draw_networkx_nodes(self.dag,  pos, nodelist=nodeList, node_color=node_color, edgecolors=edgecolors,
                               node_size=node_size, linewidths=linewidths)


    def computeStatiscalTables(self, filename='AA'):
        functionNames = []
        rts = []
        rtViolations = []
        coreAllocations = []
        for node in self.toList(self.startNodeName):
            name = node.name
            functionNames.append(name)
            # functionNames.append('NA')

            rts.append(self.AvgTotalRTs[name])  # Mean of RT
            # rts.append(self.deviationTotalRTs[name])  # Deviation of RT

            rtViolations.append(self.avgSlaViolations[name])  # % of RT violations
            # rtViolations.append(self.deviationSlaViolations[name])

            coreAllocations.append(self.avgCores[name])
            # coreAllocations.append(self.deviationCores[name])

            # rts.append(round(self.slaViolations[name]/len(self.totalRts[name])*100.0,1)) # % of RT

        # Specify the file path for the CSV file
        csv_file_path = filename + ".csv"

        # Combine the lists into a table format
        table_data = zip(functionNames, rts, rtViolations, coreAllocations)

        # Open the CSV file in write mode and specify the delimiter
        with open(csv_file_path, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",")

            # Write the header row
            writer.writerow(["Function", "RT", "Violations", "Cores"])

            # Write each row of the table data to the CSV file
            for row in table_data:
                writer.writerow(row)


    def computeFinalTable(self, simulatorLabel, timeWindow, fil1, fil2, fil3, fil4, fil5, fil6, fil7, fil8, fil9, fil10):
        with open(fil1, 'r') as f1, open(fil2, 'r') as f2, open(fil3, 'r') \
                as f3, open(fil4, 'r') as f4, open(fil5, 'r') as f5, open(fil6, 'r') \
                as f6, open(fil7, 'r') as f7, open(fil8, 'r') as f8, open(fil9, 'r') as f9, open(fil10, 'r') as f10:
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            reader3 = csv.reader(f3)
            reader4 = csv.reader(f4)
            reader5 = csv.reader(f5)
            reader6 = csv.reader(f6)
            reader7 = csv.reader(f7)
            reader8 = csv.reader(f8)
            reader9 = csv.reader(f9)
            reader10 = csv.reader(f10)

            # Skip header rows if present
            next(reader1, None)
            next(reader2, None)
            next(reader3, None)
            next(reader4, None)
            next(reader5, None)
            next(reader6, None)
            next(reader7, None)
            next(reader8, None)
            next(reader9, None)
            next(reader10, None)

            # Read the file contents into lists
            fil1 = list(reader1)
            fil2 = list(reader2)
            fil3 = list(reader3)
            fil4 = list(reader4)
            fil5 = list(reader5)
            fil6 = list(reader6)
            fil7 = list(reader7)
            fil8 = list(reader8)
            fil9 = list(reader9)
            fil10 = list(reader10)

            rts = []
            violations = []
            cores = []
            functionNames = []

            for row_num, (row1, row2, row3, row4, row5, row6, row7, row8, row9, row10) in \
                    enumerate(zip(fil1, fil2, fil3, fil4, fil5, fil6, fil7, fil8, fil9, fil10), start=2):
                if row_num > 60:
                    break

                functionNames.append(row1[0])
                functionNames.append("dev")
                # Extract values from column 2 of each row
                rt1 = float(row1[1])
                rt2 = float(row2[1])
                rt3 = float(row3[1])
                rt4 = float(row4[1])
                rt5 = float(row5[1])
                rt6 = float(row6[1])
                rt7 = float(row7[1])
                rt8 = float(row8[1])
                rt9 = float(row9[1])
                rt10 = float(row10[1])

                vio1 = float(row1[2])
                vio2 = float(row2[2])
                vio3 = float(row3[2])
                vio4 = float(row4[2])
                vio5 = float(row5[2])
                vio6 = float(row6[2])
                vio7 = float(row7[2])
                vio8 = float(row8[2])
                vio9 = float(row9[2])
                vio10 = float(row10[2])

                cor1 = float(row1[3])
                cor2 = float(row2[3])
                cor3 = float(row3[3])
                cor4 = float(row4[3])
                cor5 = float(row5[3])
                cor6 = float(row6[3])
                cor7 = float(row7[3])
                cor8 = float(row8[3])
                cor9 = float(row9[3])
                cor10 = float(row10[3])

                rtsv = [rt1, rt2, rt3, rt4, rt5, rt6, rt7, rt8, rt9, rt10]
                violationsv = [vio1, vio2, vio3, vio4, vio5, vio6, vio7, vio8, vio9, vio10]
                corsv = [cor1, cor2, cor3, cor4, cor5, cor6, cor7, cor8, cor9, cor10]
                # Calculate the sum of the values

                rts.append(round(np.mean(rtsv) + 0.0, 1))
                rts.append(round(np.std(rtsv) + 0.0, 1))
                meanViolations = np.mean(violationsv) / timeWindow*100
                deviationViolation = np.std(violationsv) / timeWindow*100
                violations.append(round(meanViolations + 0.00001, 1))
                violations.append(round(deviationViolation, 1))

                cores.append(round(np.mean(corsv) + 0.0, 1))
                cores.append(round(np.std(corsv) + 0.0, 1))

            csv_file_path = "experiments/%s-Final1.csv" % (simulatorLabel)
            # functionNames = []
            # for k in range(15):
            #     functionNames.append(f'f{k}')
            # Combine the lists into a table format
            table_data = zip(functionNames, rts, violations, cores)

            # Open the CSV file in write mode and specify the delimiter
            with open(csv_file_path, "w", newline="") as file:
                writer = csv.writer(file, delimiter=",")

                # Write the header row
                writer.writerow(["Function", "RT", "Violations", "Cores"])

                # Write each row of the table data to the CSV file
                for row in table_data:
                    writer.writerow(row)

                # Print the row sum

        # Specify the paths to file1 and file2


    def computeResultTable(self, simulatorLebal='', timeWindow=1200.0):
        file1 = "experiments/%s-1.csv" % (simulatorLebal)
        file2 = "experiments/%s-2.csv" % (simulatorLebal)
        file3 = "experiments/%s-3.csv" % (simulatorLebal)
        file4 = "experiments/%s-4.csv" % (simulatorLebal)
        file5 = "experiments/%s-5.csv" % (simulatorLebal)
        file6 = "experiments/%s-6.csv" % (simulatorLebal)
        file7 = "experiments/%s-7.csv" % (simulatorLebal)
        file8 = "experiments/%s-8.csv" % (simulatorLebal)
        file9 = "experiments/%s-9.csv" % (simulatorLebal)
        file10 = "experiments/%s-10.csv" % (simulatorLebal)
        # Call the sum_and_print_values function
        self.computeFinalTable(simulatorLebal,timeWindow, file1, file2, file3, file4, file5, file6, file7, file8, file9, file10)
