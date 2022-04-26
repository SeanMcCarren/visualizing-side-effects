import numpy as np
import pickle

class Node:
    def __init__(self, node):
        self.node = node
        self.name = int(node.name) + 0
        assert str(self.name) == node.name
        self.parents = []
        self.children = []
        self.important = False
        self.w = None
        self.d = None
        self.leafs = None

    @property
    def height(self):
        if len(self.children) == 0:
            return 0
        else:
            return max((child.height for child in self.children)) + 1

    @property
    def depth(self):
        if len(self.parents) == 0:
            return 0
        else:
            return max((parent.depth for parent in self.parents)) + 1
    
    @property
    def size(self):
        if len(self.children) == 0:
            return 1
        else:
            return sum((child.size for child in self.children)) + 1
    
    @property
    def has_important(self):
        if len(self.children) == 0:
            return self.important
        else:
            return any((child.has_important for child in self.children))
    
    def pop_child(self, node):
        if node in self.children:
            self.children.pop(self.children.index(node))
    
    def add_child(self, node):
        if node not in self.children:
            self.children.append(node)
    
    def pop_parent(self, node):
        if node in self.parents:
            self.parents.pop(self.parents.index(node))
    
    def add_parent(self, node):
        if node not in self.parents:
            self.parents.append(node)
    
    def del_subtree(self):
        for par in self.parents:
            par.pop_child(self)
        for c in self.children:
            c.del_subtree()
        del c # this does not maintain tree's concept_to_node
    
    def __repr__(self):
        return f"NODE: {self.name}. P:{len(self.parents)}, C:{len(self.children)}"

        
class Tree:
    def __init__(self, concepts, root=None):
        if root is None:
            root = Node(SNOMEDCT_US[138875005])
        self.root = root
        self.concept_to_node = {self.root.name: root}
        for concept in concepts:
            if concept is not None:
                self.expand(concept)
    
    @property
    def height(self):
        return self.root.height
    
    def expand(self, concept):
        # start higher up if we are newly making this node
        if int(concept.name) in self.concept_to_node:
            return
        for par in concept.parents:
            if int(par.name) not in self.concept_to_node:
                self.expand(par)
        node = Node(concept)
        self.concept_to_node[node.name] = node
        for par in concept.parents:
            par_node = self.concept_to_node[int(par.name)]
            if node not in par_node.children:
                par_node.children.append(node)
            if par_node not in node.parents:
                node.parents.append(par_node)
    
    def is_tree(self):
        try:
            for _ in self.traverse(yield_first_visit=False, yield_visited=True, raise_on_visited=True):
                pass
        except Exception as e:
            if e.args is not None and len(e.args) > 0 and e.args[0] == "visited":
                print(e.args[1])
                return False
            else:
                raise e
        return True
    
    def traverse(self, yield_first_visit=True, yield_visited=False, raise_on_visited=True):
        visited = set()
        to_visit = [self.root]
        while to_visit:
            n = to_visit.pop(0)
            if n in visited:
                if yield_visited:
                    yield n
                if raise_on_visited:
                    raise Exception("visited", n)
            else:
                if yield_first_visit:
                    yield n
                visited.add(n)
            to_visit.extend(n.children)

    def __attr_nr_leafs(self):
        def attr_recursive(n):
            if n.leafs is None:
                leafs = None
                if len(n.children) == 0:
                    leafs = set([n])
                else:
                    leafs = set()
                    for c in n.children:
                        leafs.update(attr_recursive(c))
                n.w = len(leafs)
                n.leafs = leafs
            return n.leafs
        attr_recursive(self.root)
        for n in self.traverse(raise_on_visited=False):
            n.leafs = None
            assert n.w != 0
    
    def attr_depth(self):
        # Always take length of shortest path to root
        for n in self.traverse(raise_on_visited=False):
            n.d = n.depth
    
    def make_tree(self, mode='most_popular_parent'):
        if mode == 'most_popular_parent':
            self.__attr_nr_leafs()
        for n in self.traverse(yield_first_visit=False, yield_visited=True, raise_on_visited=False):
            if mode == 'most_popular_parent':
                popular_parent = n.parents[0]
                for par in n.parents[1:]:
                    if par.w > popular_parent.w:
                        popular_parent.pop_child(n)
                        popular_parent = par
                    else:
                        par.pop_child(n)
                n.parents = [popular_parent]
            if mode == 'random':
                keep_i = np.random.choice(len(n.parents))
                for i, p in enumerate(n.parents):
                    if i == keep_i:
                        n.parents == [p]
                    else:
                        p.pop_child(n)
    
    def del_parent_store(self):
        for n in T.traverse(raise_on_visited=False):
            n.parents = []
        
    def add_parent_store(self):
        for n in T.traverse(raise_on_visited=False):
            for c in n.children:
                c.add_parent(n)
    
    def filter_tree(self, constraint, verbose=False):
        N = 0
        E = 0
        for n in self.traverse(yield_first_visit=True, yield_visited=False, raise_on_visited=True):
            if constraint(n):
                # reconnect
                for c in n.children:
                    for p in n.parents:
                        p.add_child(c)
                        c.add_parent(p)
                        E += 1
                # unlink parents
                for p in n.parents:
                    p.pop_child(n)
                # unlink children
                for c in n.children:
                    c.pop_parent(n)

                N += 1
                # delete node
                del self.concept_to_node[n.name]
                del n
        print(f"Removed {N} nodes and added {E} edges.")
    
    def compact(self, verbose=False):
        self.filter_tree(lambda n : len(n.children) == 1, verbose=verbose)

    def keep_concepts(self, concepts, verbose=False):
        C = 0
        for concept in concepts:
            if int(concept) not in self.concept_to_node:
                C += 1
            else:
                self.concept_to_node[int(concept)].important = True
        print(f"Warning, {C} concepts were not found in tree.")
        self.filter_tree(lambda n : not n.has_important, verbose=verbose)

def load(file='../data/pickled'):
    with open(file, 'rb') as f:
        T = pickle.load(f)
    T.add_parent_store()
    T.attr_depth()
    return T