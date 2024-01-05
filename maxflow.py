"""
  Please run this command before running this script
    pip install networkx pandas matplotlib openpyxl
"""

import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class BaseGraph:
    def __init__(self, df):
        self.df = df
        self.ROW = max(self.df['source'].max(), self.df['destination'].max()) + 1
        self.graph = None
        self._prepare_graph(df)

    def _prepare_graph(self, df):
      self.graph = defaultdict(dict)
      for index, row in df.iterrows():
        source = row['source']
        destination = row['destination']
        capacity = row['capacity']
        self.graph[source][destination] = 0
        self.graph[destination][source] = 0
      for index, row in df.iterrows():
        source = row['source']
        destination = row['destination']
        capacity = row['capacity']
        self.graph[source][destination] = capacity

    def BFS(self, source, destination, parent):
        visited = [source]
        queue = [source]
        while queue:
            u = queue.pop(0)
            for v, capacity in self.graph[u].items():
                if v not in visited and capacity > 0:
                    queue.append(v)
                    visited.append(v)
                    parent[v] = u
                    if v == destination:
                        print('visited', visited)
                        return visited
        return None

    def FordFulkerson(self, source, sink):
      traversed_paths = []
      max_flow = 0
      parent = dict()
      while visited:=self.BFS(source, sink, parent):
        traversed_paths.append(visited)
        path_flow = float("Inf")
        s = sink
        while(s !=  source):
          path_flow = min(path_flow, self.graph[parent[s]][s])
          s = parent[s]
        max_flow += path_flow
        v = sink
        while(v != source):
          u = parent[v]
          self.graph[u][v] -= path_flow
          self.graph[v][u] += path_flow
          v = parent[v]

      return max_flow, traversed_paths

class PlottableGraph(BaseGraph):
    def __init__(self, df):
      super().__init__(df)
      self.nx_graph = self._prepare_nx_graph()

    def get_directed_edges(self):
      multi_directed = []
      one_way_directed = []
      for index, row in self.df.iterrows():
        source = row['source']
        destination = row['destination']
        if (destination, source) in one_way_directed:
           multi_directed.append((source, destination))
           one_way_directed.remove((destination, source))
        else:
          one_way_directed.append((source, destination))
      return one_way_directed, multi_directed

    def _prepare_nx_graph(self):
      graph = nx.Graph()
      for index, row in self.df.iterrows():
        source = row['source']
        destination = row['destination']
        capacity = row['capacity']
        graph.add_edge(source, destination, weight=capacity)
      return graph

    def plot_graph(self, source, sink, paths, ax=None):
      visited = list(map(lambda x, y: (x, y), paths, paths[1:]))
      visited_enum = {ele:(index+1) for index, ele in enumerate(visited)}
      G = self.nx_graph
      pos = nx.spring_layout(G, seed=2394817)  # positions for all nodes - seed for reproducibility
      node_list = sorted(list(G.nodes))
      head_nodes = [source, sink]
      node_list.remove(source)
      node_list.remove(sink)
      mid_nodes = node_list 
      one_way_directed, multi_directed = self.get_directed_edges()
      # nodes
      nx.draw_networkx_nodes(G, pos, nodelist=head_nodes, node_color="tab:red", ax=ax)
      nx.draw_networkx_nodes(G, pos, nodelist=mid_nodes, node_color="tab:blue", ax=ax)
      # edges
      nx.draw_networkx_edges(
          G,
          pos,
          arrowstyle="->",
          # arrowsize=10,
          arrows=True,
          width=1,
          connectionstyle='arc',
          edgelist=one_way_directed,
          ax=ax
      )
      nx.draw_networkx_edges(
          G,
          pos,
          arrowstyle="<->",
          arrows=True,
          width=1,
          connectionstyle='arc',
          edgelist=multi_directed,
          ax=ax
      )
      nx.draw_networkx_edges(
          G,
          pos,
          arrowstyle="->",
          arrows=True,
          arrowsize=15,
          width=3,
          connectionstyle='arc',
          edgelist=visited,
          alpha=0.5,
          edge_color="tab:green",
          ax=ax
      )
      nx.draw_networkx_edge_labels(
          G,
          pos,
          edge_labels=visited_enum,
          ax=ax
      )
      # node labels
      nx.draw_networkx_labels(G, pos, ax=ax)
      ax.set_axis_off()

    def plot_graphs(self, source, sink, paths):
      if not paths:
        print(f"There is no possible way to traverse from {source} to {sink}")
        return
      fig, axes = plt.subplots(len(paths), 1, figsize=(20, len(paths)*20), dpi=100)
      for i, path in enumerate(paths):
        ax = axes[i] if type(axes) is list else axes 
        self.plot_graph(source, sink, path, ax)
        ax.set_title(f"Iteration {i+1}, traverse from {path[0]} to {path[-1]}")
      # plt.show()
      fig.savefig('maxflow.png')

if __name__ == '__main__':
  # Load data from spreadsheet
  df_dict = pd.read_excel('./btl ctrr2.xlsx', sheet_name=[1,2,3])
  raw_data = df_dict[1][["source", "destination", "capacity"]].dropna().astype(int)
  test_data = df_dict[3][["source", "destination", "capacity"]].dropna().astype(int)
  address_name = df_dict[2]
  g = PlottableGraph(raw_data)
  source = int(input('Input the source node to visit, enter 0 for default start point "Ngã tư An Sương": '))
  sink = int(input('Input the sink node to end, enter 0 for default destination "Công viên Thảo Cầm Viên": '))
  if not source:
    source = int(address_name[address_name['name'] == 'Ngã tư An Sương' ]['id'].values[0])
  if not sink:
    sink = int(address_name[address_name['name'] == 'Công viên Thảo Cầm Viên' ]['id'].values[0])
  max_flow, visited = g.FordFulkerson(source, sink)
  print ("The maximum possible flow from %s to %s is %d " % (source, sink, max_flow))
  g.plot_graphs(source, sink, visited)