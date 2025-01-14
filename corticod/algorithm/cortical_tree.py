from typing import Any, List
import numpy as np
# from bisect import bisect_right
from sklearn.metrics.pairwise import cosine_similarity

from corticod.algorithm.tree import Node, Tree
from corticod.algorithm.codebook import Codebook

NEW_DATA_WEIGHT = 0.25
MATURATION_ENERGY_COEFF = 10 # 50
MATURATION_ENERGY_THRESH = 100 # 500
DEFAULT_RANGE_INIT = 50
DEFAULT_RANGE_LIMIT = 10

class CorticalNode(Node):
    def __init__(self, range_init, parent=None):
        super().__init__(parent)
        self.maturation_energy = 0
        self.pass_count: int = 1
        self.range = range_init
        self.range_init = range_init
        
    def add_child(self, data:float, range_init:float) -> 'CorticalNode':
        new_child = CorticalNode(range_init, parent=self)
        return self.children.add_child(data, new_child)

    def update(self, dataIn:float, range_limit:float) -> bool:
        sqpc = np.power(self.pass_count, 0.5) # square root of pass count
        prev_data = self.get_data()
        temp_data = prev_data * (1 - NEW_DATA_WEIGHT) +  dataIn * NEW_DATA_WEIGHT 
        
        new_data = prev_data - prev_data/sqpc + temp_data/sqpc
        # new_data = (dataIn + sqpc*prev_data) / (sqpc + 1); //update node value
        self.set_data(new_data)

        self.range = self.range / (1 + self.level * sqpc)
        if self.range <= range_limit:
            self.range = range_limit # range cannot be less than range_limit

        self.pass_count += 1
        
        if not self.is_mature():
            error = abs(dataIn - new_data) + 1e-10 # to avoid division by zero
            energy_change = MATURATION_ENERGY_COEFF * self.level / error
            self.maturation_energy += energy_change
            if self.maturation_energy >= MATURATION_ENERGY_THRESH:
                self.set_mature(True)
                self.range = self.range_init
                return True # matured
        return False # no change

    def find_closest_child(self, data:float) -> (int, float, bool):
        
        # check only mature children first
        search_space = [np.inf if self.children.maturity_mask[i] else self.children.data_vector[i] for i in range(len(self.children.data_vector))]
        # if len(search_space) > 0:
        if sum(self.children.maturity_mask)<len(self.children.maturity_mask):
            i = np.argmin(np.abs(search_space - data))
            found_child_data = search_space[i]
            dist = abs(found_child_data - data)

            if(dist < self.children[i].range):
                return self.children[i], dist
        
        # if no mature children are close enough, check immature children
        search_space = [np.inf if not self.children.maturity_mask[i] else self.children.data_vector[i] for i in range(len(self.children.data_vector))]
        # if len(search_space) > 0:
        if sum(self.children.maturity_mask)>0:
            i = np.argmin(np.abs(search_space - data))
            found_child_data = search_space[i]
            dist = abs(found_child_data - data)

            if(dist < self.children[i].range):
                return self.children[i], dist
        
        # if no children are close enough, return None
        return None, None
    
    def find_closest_children(self, data:List[float]) -> List[int]:
        search_space = self.children.data_vector[self.children.maturity_mask]
        if len(search_space) > 0:
            dists = search_space**2 - 2 * search_space * data + data**2
            mask = dists < (self.children.range ** 2)
            return np.where(mask)[0], dists[mask]
        
        search_space = self.children.data_vector[np.bitwise_not(self.children.maturity_mask)]
        if len(search_space) > 0:
            dists = search_space**2 - 2 * search_space * data + data**2
            mask = dists < (self.children.range ** 2)
            return np.where(mask)[0], dists[mask]
        
        return [], []
        

    def get_total_progeny(self) -> int:
        total = 0
        for child in self.children:
            total += 1
            total += child.get_total_progeny()
        return total
    
    def get_level_progeny(self, target_level) -> int:
        if self.is_mature() and self.level == target_level:
            return 1
        total = 0
        for child in self.children:
            total += child.get_level_progeny(target_level)
        return total
    
    # def get_progeny_width(self) -> int:
    #     # get the widest branch along each level
    #     max_width = 0
    #     for child in self.children:
    #         width = 1 + child.get_progeny_width()
    #         if width > max_width:
    #             max_width = width
    #     return max_width


    def __repr__(self) -> str:
        log = []
        if self.parent is not None:
            log.append(f"{self.get_data():.3f}")
            log.append(f"Level: {self.level}")
        else:
            log.append("<:ROOT:>")
        log.append(f"Children: {len([c for i, c in enumerate(self.children) if self.children.maturity_mask[i]])}")
        log.append(f"Hillocks: {len([c for i, c in enumerate(self.children) if not self.children.maturity_mask[i]])}")
        log.append(f"Range: {self.range:.3f}")
        log.append(f"Pass Count: {self.pass_count}")
        # log.append(f"Maturation Energy: {self.maturation_energy:.3f}")
        log.append(f"Progeny: {self.get_total_progeny()}")
        return f"CorticalNode({', '.join(log)})"


class CortexTree(Tree):
    def __init__(self, window_size=8, range_init=DEFAULT_RANGE_INIT, range_limit=DEFAULT_RANGE_LIMIT):
        super().__init__(CorticalNode(range_init))
        self.window_size = window_size
        # self.range_limit = range_limit
        # self.range_init = range_init
        self.range_limit = [range_limit * ((0.9) ** lvl)
                            for lvl in range(window_size+1)]
        self.range_init = [range_init * ((0.9) ** lvl) 
                            for lvl in range(window_size+1)]
        self.changed = True
        self.cb = None

    def closest_path(self, wave):
        node = self.root
        for coef in wave:
            if len(node.children) > 0:
                c, c_dist = node.find_closest_child(coef)
                if c:
                    if c.is_mature():
                        node = c
                        continue
                else:
                    continue
            break
        
        path = []
        while(node.parent != None):
            path.insert(0, node.get_data())
            node = node.parent
        path = np.asarray(path)
        # dist = np.linalg.norm(path - wave[:len(path)])
        # return dist if len(path) == self.window_size else 0
        return path

    def closest_full_path(self, wave):
        if self.changed:
            paths = self.paths(self.window_size)
            if paths.ndim == 1:
                return 0
            self.cb = self.complete(paths)
        return self.cb.distance(wave)

    def train_single(self, wave):
        self.changed=True
        added = 0
        leafs = 0
        node = self.root
        for coef in wave:
            if len(node.children) > 0:
                # Search among children
                found_child, c_dist = node.find_closest_child(coef)
                if found_child is not None:
                    newly_matured = found_child.update(coef, self.range_limit[found_child.level])
                    if newly_matured: 
                        print(f"Node {found_child.get_data():.3f} matured at level {found_child.level}")
                        if found_child.level == self.window_size:
                            leafs += 1
                        added += 1
                    node = found_child
                    continue # go to next coef
            # Make new child 
            node.add_child(coef, self.range_init[node.level])
            added += 1
            break # terminate

        path = []
        while(node.parent != None):
            path.append(node.get_data())
            node = node.parent

        path = np.asarray(path)
        # dist = np.linalg.norm(path - wave[:len(path)])
        return (len(path), added, leafs)
    
    # def train_batch(self, waves):
    #     self.changed=True
    #     added = 0
    #     leafs = 0
    #     node = self.root
    #     for coefs in waves.T:
    #         found_children, c_dists = node.find_closest_children(coefs)
    #         for i, found_child in enumerate(found_children):
    #             if found_child is not None:
    #                 matured = found_child.update(coefs[i], self.range_limit[found_child.level])
    #                 if matured: 
    #                     print(f"Node matured at level {found_child.level}")
    #                     if found_child.level == self.window_size:
    #                         leafs += 1
    #                     added += 1
    #                     continue
    #                 node = found_child # this will not work for multiple children, need fix
    #                 continue
    #             node.add_child(coefs[i], self.range_init[node.level])
    #             added += 1
    #             break
    #     return (len(waves.T), added, leafs)
    
    # def train(self, waves, epochs=1, batch_size=1):
    #     added = 0
    #     leafs = 0
    #     for _ in range(epochs):
    #         for i in range(0, len(waves), batch_size):
    #             l, a, lf = self.train_batch(waves[i:i+batch_size])
    #             added += a
    #             leafs += lf
    #     return added, leafs

    def complete(self, paths=None):
        if paths is None:
            paths = self.paths(self.window_size)
        if paths.ndim == 1:
            return Codebook(np.zeros(shape=(1, self.window_size)))
        else:
            return Codebook(paths[:, :])
    