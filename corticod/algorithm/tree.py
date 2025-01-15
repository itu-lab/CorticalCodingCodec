import numpy as np

class Node():
    def __init__(self, parent:'Node'=None, **kwargs):
        self.parent = parent
        self.level = parent.level + 1 if parent is not None else 0
        self.sibling_index = None
        self.children:NodeChildren = NodeChildren(self)
    def get_data(self) -> np.ndarray:
        if self.parent is None: return None
        return self.parent.children.data_vector[self.sibling_index]
    def set_data(self, data:np.ndarray) -> None:
        if self.parent is not None:
            self.parent.children.set_data(self.sibling_index, data)
    def is_mature(self) -> bool:
        if self.parent is None: return None
        return self.parent.children.maturity_mask[self.sibling_index]
    def set_mature(self, value:bool) -> None:
        if self.parent is not None: 
            self.parent.children.maturity_mask[self.sibling_index] = value
    def add_child(self, data:np.ndarray, **kwargs) -> 'Node':
        new_child = type(self)(parent=self, **kwargs)
        return self.children.add_child(new_child, data)
    def __repr__(self):
        return f"Node({self.level}:{self.get_data()})"
    
# make another class to hold all childrens of a node, and its operations
class NodeChildren():
    def __init__(self, owner:Node, dtype:np.dtype=np.float64):
        self.owner = owner
        self.count = 0
        self.elements = []
        self.data_vector = np.array([], dtype=dtype)
        self.maturity_mask = np.array([], dtype=np.bool_)

    def find_index(self, data:np.ndarray, side:str='left') -> 'Node':
        # get the first suitable index to put the data, which will not change the ascending order
        return np.searchsorted(self.data_vector, data, side=side).item()
    
    def add_child(self, new_child:Node, data:np.ndarray, maturity:bool=False) -> Node:
        i = self.find_index(data)
        new_child.sibling_index = i
        self.elements.insert(i, new_child)
        self.data_vector = np.insert(self.data_vector, i, data)
        self.maturity_mask = np.insert(self.maturity_mask, i, maturity) # True means mature, False means not mature
        for j in range(i+1, len(self.elements)):
            self.elements[j].sibling_index += 1
        self.count += 1
        return new_child
    
    def remove_child(self, i: int):
        self.count -= 1
        for j in range(i + 1, len(self.elements)): # Adjust sibling indices for elements after the removed one
            self.elements[j].sibling_index -= 1
        removed_element = self.elements.pop(i)
        removed_data = self.data_vector[i].copy()  # Copy ensures the row is not referenced
        # removed_data = self.data_vector[i, :].copy()  # Copy ensures the row is not referenced
        self.data_vector = np.delete(self.data_vector, i, axis=0)
        removed_mask = self.maturity_mask[i].copy()  # Copy for consistency
        self.maturity_mask = np.delete(self.maturity_mask, i, axis=0)
        return removed_element, removed_data, removed_mask

    def set_data(self, i:int, data:np.ndarray) -> None:
        if self.data_vector[i] == data: # no change
            return
        if i < 0 or i >= len(self.data_vector):
            print("ERROR: Child do not exist!")
            exit(-1)
        elif len(self.data_vector) == 1: # it is certain i==0 
            self.data_vector[i] = data 
        else: # There are 2 or more children
            if (i > 0 and data < self.data_vector[i - 1]) or (i < len(self.data_vector) - 1 and data > self.data_vector[i + 1]):
                # needs arrangement
                self.add_child(*self.remove_child(i))
            else: 
                self.data_vector[i] = data 
    
    def __getitem__(self, i):
        return self.elements[i]
    def __len__(self):
        return self.count
    def __iter__(self):
        return iter(self.elements)


class Tree():
    def __init__(self, root:Node):
        self.root:Node = root
        
    def paths(self, window_size:int) -> np.ndarray:
        all_paths = []
        path = []
        def recurse(node:Node):
            if len(node.children) == 0:
                if len(path) == window_size:
                    all_paths.append(path.copy())
            else:
                for child in node.children:
                    if child.is_mature():
                        path.append(child.get_data())
                        recurse(child)
                        path.pop()

        recurse(self.root)
        return np.asarray(all_paths)