def find_best_split(feature_vector, target_vector):
    x = np.asarray(feature_vector, dtype=float)
    y = np.asarray(target_vector, dtype=int)
    n = x.shape[0]
    if n <= 1:
        return None

    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    diffs = np.diff(x_sorted)
    valid = diffs != 0
    if not np.any(valid):
        return None

    thresholds = (x_sorted[:-1] + x_sorted[1:]) / 2
    thresholds = thresholds[valid]

    cumsum_pos = np.cumsum(y_sorted)
    total_pos = cumsum_pos[-1]

    left_counts = np.arange(1, n)
    right_counts = n - left_counts

    pos_left = cumsum_pos[:-1]
    pos_right = total_pos - pos_left

    neg_left = left_counts - pos_left
    neg_right = right_counts - pos_right

    pL1 = pos_left / left_counts
    pL0 = neg_left / left_counts
    pR1 = pos_right / right_counts
    pR0 = neg_right / right_counts

    H_left = 1 - pL1**2 - pL0**2
    H_right = 1 - pR1**2 - pR0**2

    weighted_impurity = (left_counts / n) * H_left + (right_counts / n) * H_right
    ginis_all = -weighted_impurity
    ginis = ginis_all[valid]

    if ginis.size == 0:
        return None

    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._depth = 0
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]): #fixed
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and node["depth"] > self._max_depth:
          node["type"] = "terminal"
          node["class"] = Counter(sub_y).most_common(1)[0][0]
          return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
          node["type"] = "terminal"
          node["class"] = Counter(sub_y).most_common(1)[0][0]
          return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(1, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click/current_count #fix
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) #fixed
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) == 3:
                continue


            split_result = find_best_split(feature_vector, sub_y)
            if split_result is None:
                continue
            _, _, threshold, gini = split_result
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": #fix
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node['class'] = Counter(sub_y).most_common(1)[0][0] #fix
            return

        left_indices = split
        right_indices = np.logical_not(split)
    
        if self._min_samples_leaf is not None:
            if np.sum(left_indices) < self._min_samples_leaf or np.sum(right_indices) < self._min_samples_leaf:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        node["left_child"]["depth"] = node["depth"] + 1
        node["right_child"]["depth"] = node["depth"] + 1
        self._depth = max(self._depth, node["depth"] + 1)

        self._fit_node(sub_X[left_indices], sub_y[left_indices], node["left_child"])
        self._fit_node(sub_X[right_indices], sub_y[right_indices], node["right_child"])

    def _predict_node(self, x, node):
      if node["type"] == "terminal":
          return node["class"]

      feature = node["feature_split"]
      feature_type = self._feature_types[feature]

      if feature_type == "real":
          if x[feature] < node["threshold"]:
              return self._predict_node(x, node["left_child"])
          else:
              return self._predict_node(x, node["right_child"])
      elif feature_type == "categorical":
          if x[feature] in node["categories_split"]:
              return self._predict_node(x, node["left_child"])
          else:
              return self._predict_node(x, node["right_child"])
      else:
          raise ValueError("Unknown feature type")

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self._tree = {} 
        self._tree["depth"] = 1
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        X = np.array(X)
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)