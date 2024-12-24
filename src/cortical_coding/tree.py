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
        new_child = Node(self, **kwargs)
        return self.children.add_child(data, new_child)
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
        return np.searchsorted(self.data_vector, data, side=side).item()
    
    def add_child(self, data:np.ndarray, new_child:Node) -> Node:
        i = self.find_index(data)
        new_child.sibling_index = i
        self.elements.insert(i, new_child)
        self.data_vector = np.insert(self.data_vector, i, data)
        self.maturity_mask = np.insert(self.maturity_mask, i, True) # False means not mature
        for j in range(i+1, len(self.elements)):
            self.elements[j].sibling_index += 1
        self.count += 1
        return new_child
    
    def set_data(self, i:int, data:np.ndarray) -> None:
        # selected_data = self.data_vector[i]
        selected_maturity = self.maturity_mask[i]
        if i > 0 and data < self.data_vector[i - 1]: # We are moving selected to the left
            ii = self.find_index(data)
            self.data_vector[ii:i+1] = self.data_vector[ii-1:i]
            self.maturity_mask[ii:i+1] = self.maturity_mask[ii-1:i]
            self.data_vector[ii] = data
            self.maturity_mask[ii] = selected_maturity
            
            selected_element = self.elements.pop(i)
            selected_element.sibling_index = ii
            self.elements.insert(ii, selected_element)
            for j in range(ii+1, i+1):
                self.elements[j].sibling_index += 1
        elif i < len(self.data_vector) - 1 and data > self.data_vector[i + 1]: # We are moving selected to the right
            ii = self.find_index(data, side='right')
            self.data_vector[i:ii-1] = self.data_vector[i+1:ii]
            self.maturity_mask[i:ii-1] = self.maturity_mask[i+1:ii]
            self.data_vector[ii-1] = data
            self.maturity_mask[ii-1] = selected_maturity

            selected_element = self.elements.pop(i)
            selected_element.sibling_index = ii
            self.elements.insert(ii, selected_element)
            for j in range(i, ii):
                self.elements[j].sibling_index -= 1
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
    

if __name__ == "__main__":
    # Example: Organizational Hierarchy Tree
    # Root represents the CEO, children represent departments, sub-children represent teams

    # Initialize the root node (CEO)
    organization = Node()

    # add management
    ceo = organization.add_child(np.array([1000000.0])) # Salary only
    ceo.name = "CEO"

    # Add departments
    finance_dept = ceo.add_child(np.array([500000.0]))  # Salary only
    finance_dept.name = "Finance Department"

    tech_dept = ceo.add_child(np.array([700000.0]))  # Salary only
    tech_dept.name = "Tech Department"

    sales_dept = ceo.add_child(np.array([600000.0]))  # Salary only
    sales_dept.name = "Sales Department"

    # Add teams to the Finance Department
    accounting_team = finance_dept.add_child(np.array([200000.0]))  # Salary only
    accounting_team.name = "Accounting Team"

    auditing_team = finance_dept.add_child(np.array([300000.0]))  # Salary only
    auditing_team.name = "Auditing Team"

    # Add teams to the Tech Department
    dev_team = tech_dept.add_child(np.array([400000.0]))  # Salary only
    dev_team.name = "Development Team"

    it_team = tech_dept.add_child(np.array([300000.0]))  # Salary only
    it_team.name = "IT Support Team"

    # Add teams to the Sales Department
    regional_sales_team = sales_dept.add_child(np.array([350000.0]))  # Salary only
    regional_sales_team.name = "Regional Sales Team"

    online_sales_team = sales_dept.add_child(np.array([250000.0]))  # Salary only
    online_sales_team.name = "Online Sales Team"

    # Create the tree object
    org_tree = Tree(organization)

    # Example usage: Find all paths of a certain salary range
    window_size = 3  # Limit the path length to include CEO and department level
    paths = org_tree.paths(window_size)

    print("Organizational Hierarchy Paths:")
    for path in paths:
        print([p.tolist() if isinstance(p, np.ndarray) else p for p in path])

    # Example usage: Print structure and salaries
    def print_organization(node, indent=0):
        data = node.get_data()
        salary = data.item() if data is not None and isinstance(data, np.ndarray) else "N/A"
        print("  " * indent + f"{node.name}: ${salary}")
        for child in node.children:
            print_organization(child, indent + 1)

    print("\nOrganizational Hierarchy Structure:")
    print_organization(ceo)

    # Example usage: Adjust salaries
    finance_dept.set_data(np.array([550000.0]))  # Adjust salary
    print("\nUpdated Salaries:")
    print_organization(ceo)