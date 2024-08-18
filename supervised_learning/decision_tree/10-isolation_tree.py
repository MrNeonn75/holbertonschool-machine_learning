#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    '''
        Class Documentation
    '''
    def __init__(self, max_depth=10, seed=0, root=None):
        '''
            Function Documentation
        '''
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        '''
            Function Documentation
        '''
        return self.root.__str__() + "\n"

    def depth(self):
        '''
            Function Documentation
        '''
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        '''
            Function Documentation
        '''
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        '''
            Function Documentation
        '''
        self.root.update_bounds_below()

    def get_leaves(self):
        '''
            Function Documentation
        '''
        return self.root.get_leaves_below()

    def update_predict(self):
        '''
            Function Documentation
        '''
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            '''
                Function Documentation
            '''
            predictions = np.zeros(A.shape[0], dtype=int)
            for i, x in enumerate(A):
                for leaf in leaves:
                    if leaf.indicator(np.array([x])):
                        predictions[i] = leaf.value
                        break
            return predictions
        self.predict = predict

    def np_extrema(self, arr):
        '''
            Function Documentation
        '''
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        '''
            Function Documentation
        '''
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(self.explanatory
                                                       [:, feature]
                                                       [node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        '''
            Function Documentation
        '''
        value = node.depth + 1
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        '''
            Function Documentation
        '''
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        '''
            Function Documentation
        '''
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & \
            (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population

        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop)
        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            node.left_child.depth = node.depth + 1
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            node.right_child.depth = node.depth + 1
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        '''
            Function Documentation
        '''
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
