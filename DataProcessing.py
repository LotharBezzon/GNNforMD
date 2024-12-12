import numpy as np
from torch_geometric.data import Data
import torch
import torch.nn.functional as Func

def read_data(files, molecular=True):
    """
    Reads data from a list of LAMMPS trajectory files and extracts atom information.

    Args:
        files (list of str): List of file paths to LAMMPS trajectory files.

    Returns:
        list of dict: A list of dictionaries, each containing information about a frame.
                      Each dictionary has the following keys:
                      - 'timestep': The timestep of the frame.
                      - 'num_atoms': The number of atoms in the frame.
                      - 'box_size': The size of the simulation box.
                      - 'atoms': A list of dictionaries, each containing information about an atom.
                                Each atom dictionary has the following keys:
                                - 'id': Atom ID (0-based indexing).
                                - 'mol': Molecule ID.
                                - 'type': Atom type.
                                - 'x': x-coordinate.
                                - 'y': y-coordinate.
                                - 'z': z-coordinate.
                                - 'fx': Force in the x-direction.
                                - 'fy': Force in the y-direction.
                                - 'fz': Force in the z-direction.
    """
    data = []
    for file in files:
        with open(file, 'r') as f:
            lines = iter(f.readlines())
            file_data = []
            for line in lines:
                if line.startswith('ITEM: TIMESTEP'):
                    timestep = int(next(lines))
                    next(lines)  # Skip ITEM: NUMBER OF ATOMS
                    atom_number = int(next(lines))
                    next(lines)  # Skip ITEM: BOX BOUNDS
                    bound_x = next(lines).split()
                    size_x = - float(bound_x[0]) + float(bound_x[1])
                    bound_y = next(lines).split()
                    size_y = - float(bound_y[0]) + float(bound_y[1])
                    bound_z = next(lines).split()
                    size_z = - float(bound_z[0]) + float(bound_z[1])
                    box_size = torch.tensor([size_x, size_y, size_z], dtype=torch.float)
                    next(lines)  # Skip ITEM: ATOMS id mol type x y z fx fy fz
                    atoms = []
                    for _ in range(atom_number):
                        atom_data = next(lines).split()
                        if molecular:
                            i = 0
                        else:
                            i = 1
                        atom = {
                            'id': int(atom_data[0]) - 1, # 0-based indexing
                            'mol': int(atom_data[1-i]), # if molecular mol id equal atom id
                            'type': int(atom_data[2-i]),
                            'x': float(atom_data[3-i]),
                            'y': float(atom_data[4-i]),
                            'z': float(atom_data[5-i]),
                            'fx': float(atom_data[6-i]),
                            'fy': float(atom_data[7-i]),
                            'fz': float(atom_data[8-i])
                        }
                        atoms.append(atom)
                    file_data.append({'timestep': timestep, 'num_atoms': atom_number, 'box_size': box_size, 'atoms': atoms})
        data += file_data
    return data

# To account for periodic boundary conditions
def minimum_image_distance(coords1, coords2, box_size):
    """
    Calculate the minimum image distance between two sets of coordinates considering periodic boundary conditions. Coordinates are assued to be cartesian!

    Args:
        coords1 (torch.Tensor): Tensor of shape (N, 3) representing the first set of coordinates.
        coords2 (torch.Tensor): Tensor of shape (N, 3) representing the second set of coordinates.
        box_size (torch.Tensor or float): Size of the periodic box.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) representing the three components of distances between each pair of coordinates.
    """
    # Check if the input shapes are the same
    if coords1.shape != coords2.shape:
        raise ValueError(f"Shape mismatch: coords1 has shape {coords1.shape}, but coords2 has shape {coords2.shape}")
    
    delta = coords1 - coords2
    delta -= torch.round(delta / box_size) * box_size
    distance = torch.norm(delta, dim=-1)
    return delta, distance

# Build graphs for a SchNet architecture
def make_graphs(data, charges, LJ_params, cutoff):
    """
    Build graphs for a GNN architecture from the given data.
    Nodes represent atoms and their feature is the atom type. Nodes closer than 2.3 are connected.
    Edge attributes contain the distance between the two atoms and informations about the bond type.
    The targets are the three force components acting on each atom. The forces are normalized.

    Args:
        data (list of dict): List of dictionaries, each containing information about a frame.
                             Each dictionary has the following keys:
                             - 'timestep': The timestep of the frame.
                             - 'num_atoms': The number of atoms in the frame.
                             - 'box_size': The size of the simulation box.
                             - 'atoms': A list of dictionaries, each containing information about an atom.
                                       Each atom dictionary has the following keys:
                                       - 'id': Atom ID (0-based indexing).
                                       - 'mol': Molecule ID.
                                       - 'type': Atom type.
                                       - 'x': x-coordinate.
                                       - 'y': y-coordinate.
                                       - 'z': z-coordinate.
                                       - 'fx': Force in the x-direction.
                                       - 'fy': Force in the y-direction.
                                       - 'fz': Force in the z-direction.

    Returns:
        list of torch_geometric.data.Data: List of graph data objects for the GNN architecture.
    """
    # Check if all frames have the same number of atoms
    num_atoms = data[0]['num_atoms']
    for frame in data:
        if frame['num_atoms'] != num_atoms:
            raise ValueError(f"Frame at timestep {frame['timestep']} has a different number of atoms: {frame['num_atoms']} (expected {num_atoms}) \
                             Simulations with different numbers of atoms should be handled separately and than added.")

    graphs = []
    forces_list = []
    for frame in data:
        atoms = frame['atoms']
        x = torch.tensor([[charges[atom['type']], LJ_params[atom['type']][0], LJ_params[atom['type']][1]] for atom in atoms], dtype=torch.float)

        # Create a fully connected graph
        edge_index = torch.combinations(torch.arange(frame['num_atoms']), r=2).t()
        
        # Duplicate edges in the opposite direction to make it undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        positions = torch.tensor([[atom['x'], atom['y'], atom['z']] for atom in atoms], dtype=torch.float)
        start_nodes, end_nodes = edge_index
        dist_coords, dist_mod = minimum_image_distance(positions[start_nodes], positions[end_nodes], frame['box_size'])
        
        # Filter out edges with distances greater than 2.3
        mask = dist_mod < cutoff
        edge_index = edge_index[:, mask]
        dist_mod = dist_mod[mask]
        dist_coords = dist_coords[mask]

        atom_mols = torch.tensor([atom['mol'] for atom in atoms])
        atom_types = torch.tensor([atom['type'] for atom in atoms])
        
        edge_attr = torch.zeros((edge_index.size(1), 7), dtype=torch.float)
        mol_diff = atom_mols[edge_index[0]] != atom_mols[edge_index[1]]
        type_diff = atom_types[edge_index[0]] != atom_types[edge_index[1]]
        
        edge_attr[mol_diff, 0] = 1  # LJ + Coul for distant atoms
        edge_attr[~mol_diff & type_diff, 1] = 1  # OH bonds
        edge_attr[~mol_diff & ~type_diff, 2] = 1  # HOH angle rigidity
        edge_attr[:, 3] = dist_mod
        edge_attr[:, 4:7] = dist_coords / dist_mod.view(-1,1)

        force = torch.tensor([[atom['fx'], atom['fy'], atom['fz']] for atom in atoms], dtype=torch.float)

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        forces_list.append(force)
    forces = torch.stack(forces_list)
    #forces_mean = forces.mean(dim=0, keepdim=True)
    #forces_std = forces.std(dim=0, keepdim=True)
    #forces = (forces - forces_mean) / forces_std
    for i, graph in enumerate(graphs):
        graph.y = forces[i]
    return graphs

