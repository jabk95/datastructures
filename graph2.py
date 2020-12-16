from collections import deque
import math
from disjointsets import DisjointSets
from pq import PQ
import random
import timeit
import pandas
# Programming Assignment 3
# (5) After doing steps 1 through 4 below (look for relevant comments), return up here.
#     Given the output of step 4, how do the 2 versions of Dijkstra's algorithm compare?
#     How does graph density affect performance?  Does size of the graph otherwise affect performance?
#     Any other observations?



def generate_random_weighted_digraph(v,e,min_w,max_w) :
    """Generates and returns a random weighted directed graph with v vertices and e different edges.

    Keyword arguments:
    v - number of vertices
    e - number of edges
    min_w - minimum weight
    max_w - maximum weight
    """

    # Programming Assignment 3
    # (1) Implement this function, which should:
    #    -- Generate a random weighted directed graph with v vertices and e different edges.
    #    -- Start by generating a list of random edges (assume vertices numbered from 0 to v-1).
    #       In this list, each edge is a 2-tuple, e.g., (2, 3) is an edge from vertex 2 to vertex 3.
    #    -- Next, generate a list, the same length, of random weights, with minW and maxW specifying the
    #       range of weights.
    #    -- Then construct a Digraph object passing your lists above as parameters.  The Digraph class extends
    #       Graph so actually has the same __init__ parameters as its parent class.
    #    -- return that Digraph object
    ver = v-1
    edges = []
    edge_set = set()
    i = 0
    while i < e:
        a = random.randint(0, ver)
        b = random.randint(0, ver)
        edge = (a, b)
        if a!=b and edge not in edge_set:
            edge_set.add(edge)
            edges.append(edge)
            i = i + 1

    weights = []
    for i in range(e):
        weight = (random.randint(min_w, max_w))
        weights.append(weight)
    graph = Digraph(v, edges, weights)
    return graph


    #pass



def time_shortest_path_algs() :
    """Generates a table of timing results comparing two versions of Dijkstra's algorithm."""

    # Programming Assignment 3
    #
    # (4) Make sure you find steps 2 and 3 later in this module (laster in this file in the Digraph class) then
    #     return up here to finish assignment.
    #     Implement the following function to do the following:
    #     -- Use your function from step (1) generate a random weighted directed graph with 16 vertices and 240 edges
    #        (i.e., completely connected--all possible directed edges) and weights random in interval 1 to 10 inclusive.
    #     -- Read documentation of timeit (https://docs.python.org/3/library/timeit.html)
    #     -- Use timeit to time both versions of Dijkstra that you implemented in steps 2 and 3 on this graph.  The number parameter
    #        to timeit controls how many times the thing you're timing is called.  To get meaningful times, you will need to experiment with this
    #        a bit.  E.g., increase it if the times are too small.  Use the same value of number for timing both algorithms.
    #     -- Now repeat this for a digraph with 64 vertices and 4032 edges.
    #     -- Now repeat this for a digraph with 256 vertices and 65280 edges.
    #     -- Now repeat for 16 vertices and 60 edges.
    #     -- Now repeat for 64 vertices and 672 edges.
    #     -- Now repeat for 256 vertices and 8160 edges.
    #     -- Repeat this again for 16 vertices and 32 edges.
    #     -- Repeat yet again with 64 vertices and 128 edges.
    #     -- Repeat yet again with 256 vertices and 512 edges.
    #
    #     -- Have this function output the timing data in a table, with columns for number of vertices, number of edges, and time.
    #     -- If you want, you can include larger digraphs.  The pattern I used when indicating what size to use: Dense graphs: v, e=v*(v-1),
    #        Sparse: v, e=2*v, and Something in the middle: v, e=v*(v-1)/lg V.


    from __main__ import generate_random_weighted_digraph
    values = [
                (16, 240),
                (64, 4032),
                (256, 65280),
                (16, 60),
                (64, 672),
                (256, 8160),
                (16, 32),
                (64, 128),
                (256, 512)
                ]
    global graphs
    graphs= []
    for i in range(len(values)):
        graphs.append(generate_random_weighted_digraph(values[i][0], values[i][1], 1, 10))

    time = [(timeit.timeit(stmt = 'd.dijkstras_version_1(0)', globals = {'d': d}, number= 100),
        timeit.timeit(stmt = 'd.dijkstras_version_2(0)', globals = {'d': d}, number= 100)) for d in graphs   ]
    times= []

    for i in range(9):
        times.append((values[i][0], values[i][1], time[i][0],  time[i][1]))


    table = pandas.DataFrame(times, columns = [ "vertices", "edges", "Dijkstra_1", "Dijkstra_2"], dtype = 'float64')
    print(table)


class Graph :
    """Graph represented with adjacency lists."""

    __slots__ = ['_adj']

    def __init__(self, v=10, edges=[], weights=[]) :
        """Initializes a graph with a specified number of vertices.

        Keyword arguments:
        v - number of vertices
        edges - any iterable of ordered pairs indicating the edges
        weights - (optional) list of weights, same length as edges list
        """

        self._adj = [ _AdjacencyList() for i in range(v) ]
        i=0
        hasWeights = len(edges)==len(weights)
        for a, b in edges :
            if hasWeights :
                self.add_edge(a,b,weights[i])
                i = i + 1
            else :
                self.add_edge(a, b)



    def add_edge(self, a, b, w=None) :
        """Adds an edge to the graph.

        Keyword arguments:
        a - first end point
        b - second end point
        w - weight for the edge (optional)
        """

        self._adj[a].add(b, w)
        self._adj[b].add(a, w)


    def num_vertices(self) :
        """Gets number of vertices of graph."""

        return len(self._adj)


    def degree(self, vertex) :
        """Gets degree of specified vertex.

        Keyword arguments:
        vertex - integer id of vertex
        """

        return self._adj[vertex]._size

    def bfs(self, s) :
        """Performs a BFS of the graph from a specified starting vertex.
        Returns a list of objects, one per vertex, containing the vertex's distance
        from s in attribute d, and vertex id of its predecessor in attribute pred.

        Keyword arguments:
        s - the integer id of the starting vertex.
        """

        class VertexData :
            __slots__ = [ 'd', 'pred' ]

            def __init__(self) :
                self.d = math.inf
                self.pred = None

        vertices = [VertexData() for i in range(len(self._adj))]
        vertices[s].d = 0
        q = deque([s])
        while len(q) > 0 :
            u = q.popleft()
            for v in self._adj[u] :
                if vertices[v].d == math.inf :
                    vertices[v].d = vertices[u].d + 1
                    vertices[v].pred = u
                    q.append(v)
        return vertices

    def dfs(self) :
        """Performs a DFS of the graph.  Returns a list of objects, one per vertex, containing
        the vertex's discovery time (d), finish time (f), and predecessor in the depth first forest
        produced by the search (pred).
        """

        class VertexData :
            __slots__ = [ 'd', 'f', 'pred' ]

            def __init__(self) :
                self.d = 0
                self.pred = None

        vertices = [VertexData() for i in range(len(self._adj))]
        time = 0

        def dfs_visit(u) :
            nonlocal time
            nonlocal vertices

            time = time + 1
            vertices[u].d = time
            for v in self._adj[u] :
                if vertices[v].d == 0 :
                    vertices[v].pred = u
                    dfs_visit(v)
            time = time + 1
            vertices[u].f = time

        for u in range(len(vertices)) :
            if vertices[u].d == 0 :
                dfs_visit(u)
        return vertices


    def print_graph(self, with_weights=False) :
        """Prints the graph."""

        for v, vList in enumerate(self._adj) :
            print(v, end=" -> ")
            if with_weights :
                for u, w in vList.__iter__(True) :
                    print(u, "(" + str(w) + ")", end="\t")
            else :
                for u in vList :
                    print(u, end="\t")
            print()

    def get_edge_list(self, with_weights=False) :
        """Returns a list of the edges of the graph
        as a list of tuples.  Default is of the form
        [ (a, b), (c, d), ... ] where a, b, c, d, etc are
        vertex ids.  If with_weights is True, the generated
        list includes the weights in the following form
        [ ((a, b), w1), ((c, d), w2), ... ] where w1, w2, etc
        are the edge weights.

        Keyword arguments:
        with_weights -- True to include weights
        """

        edges = []
        for v, vList in enumerate(self._adj) :
            if with_weights :
                for u, w in vList.__iter__(True) :
                    edges.append(((v,u),w))
            else :
                for u in vList :
                    edges.append((v,u))
        return edges

    def mst_kruskal(self) :
        """Returns the set of edges in some
        minimum spanning tree (MST) of the graph,
        computed using Kruskal's algorithm.
        """

        A = set()
        forest = DisjointSets(len(self._adj))
        edges = self.get_edge_list(True)
        edges.sort(key=lambda x : x[1])
        for e, w in edges :
            if forest.find_set(e[0]) != forest.find_set(e[1]) :
                A.add(e)
                #A = A | {e}
                forest.union(e[0],e[1])
        return A


    def mst_prim(self, r=0) :
        """Returns the set of edges in some
        minimum spanning tree (MST) of the graph,
        computed using Prim's algorithm.

        Keyword arguments:
        r - vertex id to designate as the root (default is 0).
        """

        parent = [ None for x in range(len(self._adj))]
        Q = PQ()
        Q.add(r, 0)
        for u in range(len(self._adj)) :
            if u != r :
                Q.add(u, math.inf)
        while not Q.is_empty() :
            u = Q.extract_min()
            for v, w in self._adj[u].__iter__(True) :
                if Q.contains(v) and w < Q.get_priority(v) :
                    parent[v] = u
                    Q.change_priority(v, w)
        A = set()
        for v, u in enumerate(parent) :
            if u != None :
                A.add((u,v))
                #A = A | {(u,v)}
        return A




class Digraph(Graph) :

    def __init__(self, v=10, edges=[], weights=[]) :
        super(Digraph, self).__init__(v, edges, weights)

    def add_edge(self, a, b, w=None) :
        self._adj[a].add(b, w)


    def dijkstras_version_1(self,s) :
        """Dijkstra's Algorithm using a simple list as the PQ."""

        # Programming Assignment 3:
        # 2) Implement Dijkstra's Algorithm using a simple list as the "priority queue" as described in paragraph
        #    that starts at bottom of page 661 and continues on 662 (also described in class).
        #
        #    Have this method return a list of 3-tuples, one for each vertex, such that first position is vertex id,
        #    second is distance from source vertex (i.e., what pseudocode from textbook refers to as v.d), and third
        #    is the vertex's parent (what the textbook refers to as v.pi).  E.g., (2, 10, 5) would mean the shortest path
        #    from s to 2 has weight 10, and vertex 2's parent is vertex 5.
        #
        #    the parameter s is the source vertex.

        def extract_min(self, s):
            min = math.inf
            index = 0
            for i in range(len(self)):
                if i not in s and self[i] < min:
                    min = self[i]
                    index = i
            return min, index
        print(self._adj[0])
        parent = [ None for x in self._adj]


        Q = []
        S = []
        for i in range(len(self._adj)) :
            Q.append(math.inf)

        Q[s] = 0
        A = []
        while  len(Q) != len(S) :
            weight, u = extract_min(Q, A)
            A.append(u)
            S.append((u, weight, parent[u]))
            for v, w in self._adj[u].__iter__(True):
                alt = weight + w
                if v not in A and alt < Q[v] :
                    parent[v] = u
                    Q[v] = alt

        return  S





    def dijkstras_version_2(self,s) :
        """Dijkstra's Algorithm using a binary heap as the PQ."""

        # Programming Assignment 3:
        # 3) Implement Dijkstra's Algorithm using a binary heap implementation of a PQ as the PQ.
        #    Specifically, use the implementation I have posted here: https://github.com/cicirello/PythonDataStructuresLibrary
        #    Use the download link (if you simply click pq.py Github will just show you the source in a web browser with line numbers).
        #
        #    Have this method return a list of 3-tuples, one for each vertex, such that first position is vertex id,
        #    second is distance from source vertex (i.e., what pseudocode from textbook refers to as v.d), and third
        #    is the vertex's parent (what the textbook refers to as v.pi).  E.g., (2, 10, 5) would mean the shortest path
        #    from s to 2 has weight 10, and vertex 2's parent is vertex 5.
        #
        #    the parameter s is the source vertex.
        parent = [ None for x in self._adj]

        Q = PQ()
        Q.add(s, 0)
        S = []
        for u in range(len(self._adj)) :
            if u != s :
                Q.add(u, math.inf)
        while not Q.is_empty() :
            u = Q.peek_min()
            distance = Q.get_priority(u)
            u =  Q.extract_min()
            S.append((u, distance, parent[u]))
            for v, w in self._adj[u].__iter__(True) :
                alt = distance + w

                if  Q.contains(v) and alt < Q.get_priority(v):
                    parent[v] = u

                    Q.change_priority(v, alt)

        return S



    def dijkstras_version_3(self,s) :
        # Programming Assignment 3:
        # EXTRA CREDIT 6: Make sure you do all required parts, assignment steps 1 to 5, before
        # even considering doing this optional extra credit part.
        #
        # Implement Dijkstra's Algorithm using a fibonacci heap implementation of a PQ as the PQ.
        # To do this, you must first implement a fibonacci heap (see the textbook chapter on fibonacci
        # heaps, Chapter 19).  You are not allowed to use an existing fibonacci heap class found from
        # the internet.  If you do the extra credit, you must implement your own.  Put your fibonacci heap class
        # in a separate .py file and make sure you include it with your homework submission.
        # Your dijkstras algorithm implementation here that uses it should have the same return
        # type as the previous two versions from steps 2 and 3 of the assignment.
        # If you do the extra credit portion, then modify your step 4 to include this version in the
        # timing results.
        #
        # This is worth very substantial extra credit if you do it, and if you do it correctly.  Implementing
        # the fibonacci heap is at least as hard (probably harder) than the first 3 programming assignments
        # combined.
        pass


    def topological_sort(self) :
        """Topological Sort of the directed graph (Section 22.4 from textbook).
        Returns the topological sort as a list of vertex indices.
        """



    def transpose(self) :
          """Computes the transpose of a directed graph. (See textbook page 616 for description of transpose).
        Does not alter the self object.  Returns a new Digraph that is the transpose of self."""

    def strongly_connected_components(self) :
        """Computes the strongly connected components of a digraph.
        Returns a list of lists, containing one list for each strongly connected component,
        which is simply a list of the vertices in that component."""



class _AdjacencyList :

    __slots__ = [ '_first', '_last', '_size']

    def __init__(self) :
        self._first = self._last = None
        self._size = 0

    def add(self, node, w=None) :
        if self._first == None :
            self._first = self._last = _AdjListNode(node, w)
        else :
            self._last._next = _AdjListNode(node, w)
            self._last = self._last._next
        self._size = self._size + 1

    def __iter__(self, weighted=False):
        if weighted :
            return _AdjListIterWithWeights(self)
        else :
            return _AdjListIter(self)





class _AdjListNode :

    __slots__ = [ '_next', '_data', '_w' ]

    def __init__(self, data, w=None) :
        self._next = None
        self._data = data
        self._w = w



class _AdjListIter :

    __slots__ = [ '_next', '_num_calls' ]

    def __init__(self, adj_list) :
        self._next = adj_list._first
        self._num_calls = adj_list._size

    def __iter__(self) :
        return self

    def __next__(self) :
        if self._num_calls == 0 :
            raise StopIteration
        self._num_calls = self._num_calls - 1
        data = self._next._data
        self._next = self._next._next
        return data

class _AdjListIterWithWeights :

    __slots__ = [ '_next', '_num_calls' ]

    def __init__(self, adj_list) :
        self._next = adj_list._first
        self._num_calls = adj_list._size

    def __iter__(self) :
        return self

    def __next__(self) :
        if self._num_calls == 0 :
            raise StopIteration
        self._num_calls = self._num_calls - 1
        data = self._next._data
        w = self._next._w
        self._next = self._next._next
        return data, w




if __name__ == "__main__" :
    d = generate_random_weighted_digraph(16, 240, 1, 10)
    print(d.dijkstras_version_1(0))
    #print(d.dijkstras_version_2(0))
    # here is where you will implement any code necessary to confirm that your
    # methods work correctly.
    # Code in this if block will only run if you run this module, and not if you load this module with
    # an import for use by another module.
    #time_shortest_path_algs()
    q = [1,2, 3]
    q[0].append(1)
