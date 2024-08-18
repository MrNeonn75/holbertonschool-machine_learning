#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np
from scipy import stats
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    '''
        Class Documentation
    '''
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        '''
            Function Documentation
        '''
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        '''
            Function Documentation
        '''
        all_preds = []
        for tree_predict in self.numpy_preds:
            preds = tree_predict(explanatory)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        mode_preds = stats.mode(all_preds, axis=0)[0]
        return mode_preds.flatten()

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        '''
            Function Documentation
        '''
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop, seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}""")
            print(f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, test_explanatory, test_target):
        '''
            Function Documentation
        '''
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
