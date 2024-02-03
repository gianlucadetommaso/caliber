import numpy as np
from sklearn.tree import _tree


def get_tree_partitions(tree, inputs: np.ndarray) -> np.ndarray:
    features = tree.tree_.feature
    left_children = tree.tree_.children_left
    right_children = tree.tree_.children_right
    thresholds = tree.tree_.threshold

    all_conds = []

    def recurse(node, conds):
        if features[node] != _tree.TREE_UNDEFINED:
            recurse(
                left_children[node],
                conds * (inputs[:, features[node]] <= thresholds[node]),
            )
            recurse(
                right_children[node],
                conds * (inputs[:, features[node]] > thresholds[node]),
            )
        else:
            all_conds.append(conds)

    recurse(0, np.ones(inputs.shape[0]).astype(bool))
    return np.stack(all_conds, axis=1)
