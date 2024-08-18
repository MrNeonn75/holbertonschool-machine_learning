#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


def left_child_add_prefix(text):
    '''
        Function Documentation
    '''
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |  "+x) + "\n"
    return new_text


def right_child_add_prefix(text):
    '''
        Function Documentation
    '''
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("       " + x) + "\n"
    return new_text


class Node:
    '''
        Class Documentation
    '''
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        '''
            Function Documentation
        '''
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        '''
            Function Documentation
        '''
        if self.is_leaf:
            return self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        else:
            left_depth = self.depth
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        else:
            right_depth = self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        '''
            Function Documentation
        '''
        if self.is_leaf:
            return 1
        if self.left_child:
            left_count = self.left_child.count_nodes_below(only_leaves)
        else:
            left_count = 0
        if self.right_child:
            right_count = self.right_child.count_nodes_below(only_leaves)
        else:
            right_count = 0
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def __str__(self):
        '''
            Function Documentation
        '''
        if self.is_root:
            Type = "root "
        elif self.is_leaf:
            return f"-> leaf [value={self.value}]"
        else:
            Type = "-> node "
        if self.left_child:
            left_str = left_child_add_prefix(str(self.left_child))
        else:
            left_str = ""
        if self.right_child:
            right_str = right_child_add_prefix(str(self.right_child))
        else:
            right_str = ""
        return f"{Type}[feature={self.feature}, threshold=\
{self.threshold}]\n{left_str}{right_str}".rstrip()

    def get_leaves_below(self):
        '''
            Function Documentation
        '''
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        '''
            Function Documentation
        '''
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        '''
            Function Documentation
        '''

        def is_large_enough(x):
            '''
                Function Documentation
            '''
            return np.all(np.array([x[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            '''
                Function Documentation
            '''
            return np.all(np.array([x[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: \
            np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        '''
            Function Documentation
        '''
        if self.is_leaf:
            return self.value
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    '''
        Class Documentation
    '''
    def __init__(self, value, depth=None):
        '''
            Function Documentation
        '''
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        '''
            Function Documentation
        '''
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        '''
            Function Documentation
        '''
        return 1

    def __str__(self):
        '''
            Function Documentation
        '''
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        '''
            Function Documentation
        '''
        return [self]

    def update_bounds_below(self):
        '''
            Function Documentation
        '''
        pass

    def pred(self, x):
        '''
            Function Documentation
        '''
        return self.value


class Decision_Tree():
    '''
        Class Documentation
    '''
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        '''
            Function Documentation
        '''
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

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

    def __str__(self):
        '''
            Function Documentation
        '''
        return self.root.__str__() + "\n"

    def get_leaves(self):
        '''
            Function Documentation
        '''
        return self.root.get_leaves_below()

    def update_bounds(self):
        '''
            Function Documentation
        '''
        self.root.update_bounds_below()

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

    def pred(self, x):
        '''
            Function Documentation
        '''
        return self.root.pred(x)

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

    def fit(self, explanatory, target, verbose=0):
        '''
            Function Documentation
        '''
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        '''
            Function Documentation
        '''
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & \
            (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population
        if len(left_population) != len(self.target):
            left_population = np.pad(left_population,
                                     (0, len(self.target) -
                                      len(self.left_population)),
                                     'constant', constant_values=(0))
        if len(right_population) != len(self.target):
            right_population = np.pad(right_population,
                                      (0, len(self.target) -
                                       len(self.right_population)),
                                      'constant', constant_values=(0))
        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop or
                        np.unique(self.target[left_population]).size == 1)
        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop or
                         np.unique(self.target[right_population]).size == 1)
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

    def get_leaf_child(self, node, sub_population):
        '''
            Function Documentation
        '''
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        '''
            Function Documentation
        '''
        A = self.target[sub_population]
        B, C = np.unique(A, return_counts=True)
        value = B[np.argmax(C)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        '''
            Function Documentation
        '''
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        '''
            Function Documentation
        '''
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target))/test_target.size
