import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from DataProcessing import read_data, make_graphs
from models import GNN, GATModel
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rotation_matrix(yaw, pitch, roll):
    """
    Generate a 3D rotation matrix using Euler angles.

    Args:
        yaw (float): yaw Euler angle.
        pitch (float): pitch Euler angle.
        roll (float): roll Euler angle.
    
    Returns:
        torch.Tensor: A 3x3 rotation matrix.
    """

    # Compute rotation matrices for each axis
    R_z = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ]).squeeze()

    R_y = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ]).squeeze()

    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ]).squeeze()

    # Combine the rotation matrices
    R = R_z @ R_y @ R_x
    return R

def rotate_graph(data, yaw, pitch, roll):
    """
    Rotate the positions of nodes in a graph using a rotation matrix.
    
    Args:
        data (torch_geometric.data.Data): A graph data object.
        yaw (float): yaw Euler angle.
        pitch (float): pitch Euler angle.
        roll (float): roll Euler angle.
    
    Returns:
        torch_geometric.data.Data: The rotated graph data object.
    """
    R = rotation_matrix(yaw, pitch, roll)
    edge_attr = torch.cat((data.edge_attr[:, :4], torch.matmul(data.edge_attr[:, 4:], R.T)), dim=1)
    y = torch.matmul(data.y, R.T)
    return Data(x=data.x, edge_index=data.edge_index, edge_attr=edge_attr, y=y)

class my_loss(torch.nn.Module):
    """
    Custom loss function module that combines L1 loss with a regularization loss.

    Methods:
        forward(pred, target, atom_num, batch_size): Computes the combined loss.

    Args:
        pred (Tensor): Predicted values.
        target (Tensor): Target values.
        atom_num (int): Number of atoms in each graph.
        batch_size (int): Number of graphs in the batch.

    Returns:
        Tensor: The computed loss value.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, atom_num, batch_size):
        return torch.F.l1_loss(pred, target, reduction='sum') + regularization_loss(pred, atom_num, batch_size)

def regularization_loss(pred, atom_num, batch_size):
    """
    Computes a loss proportional to the total net force on the system.

    Args:
        pred (Tensor): Predicted values.
        atom_num (int): Number of atoms in each graph.
        batch_size (int): Number of graphs in the batch.

    Returns:
        Tensor: The computed regularization loss value.
    """
    loss = 0
    for i in range(batch_size):     # Needed to 'unbatch' the graphs
        loss += torch.norm(torch.sum(pred[i*atom_num:(i+1)*atom_num], dim=0))
    return loss * np.sqrt(atom_num)

def train(model, optimizer, loader, lossFunc):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loader (DataLoader): DataLoader providing the training data.
        lossFunc (callable): The loss function to use.

    Returns:
        float: The average loss over the training dataset.
    """
    model.train()
    total_loss = 0
    global batch_size
    for data in loader:
        data = data.to(device)
        data = rotate_graph(data, torch.rand(1)*2*np.pi, torch.rand(1)*2*np.pi, torch.rand(1)*2*np.pi)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Forward pass.
        loss = lossFunc(out, data.y)  # Loss computation.

        loss.backward()  # Backward pass.
        
        optimizer.step()  # Update model parameters.
        total_loss += loss.item()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader, lossFunc):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader providing the test data.
        lossFunc (callable): The loss function to use.

    Returns:
        float: The average loss over the test dataset.
    """
    model.eval()
    total_loss = 0
    total_lossx = 0
    total_lossy = 0
    total_lossz = 0
    count = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = lossFunc(pred, data.y)
        lossx = lossFunc(pred[:, 0], data.y[:, 0])
        lossy = lossFunc(pred[:, 1], data.y[:, 1])
        lossz = lossFunc(pred[:, 2], data.y[:, 2])
        count += 1
        if count % 3 == 0:
            print(pred[:5], data.y[:5])
        total_loss += loss.item()
        total_lossx += lossx.item()
        total_lossy += lossy.item()
        total_lossz += lossz.item()
    return total_loss / len(loader.dataset), total_lossx / len(loader.dataset), total_lossy / len(loader.dataset), total_lossz / len(loader.dataset)

def save_checkpoint(model, optimizer, epoch, checkpoint_dir='checkpoints'):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        checkpoint_dir (str, optional): Directory to save the checkpoint. Default is 'checkpoints'.

    Returns:
        None
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'big_cutoff_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load.
        optimizer (torch.optim.Optimizer): The optimizer to load.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        int: The epoch number from the loaded checkpoint.
    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Checkpoint loaded from epoch {epoch}')
        return epoch
    else:
        print(f'No checkpoint found at {checkpoint_path}')
        return 0

if __name__ == '__main__':
    charges = [None, -0.82, 0.41]
    LJ_params = [None, (0.155, 3.165), (0, 0)]
    files = [f'data/N216.{i}.lammpstrj' for i in range(1, 11)]
    data = read_data(files)
    print('Data read')
    
    graphs = make_graphs(data, charges, LJ_params, cutoff=3.5)
    print(len(graphs))
    print('Graphs made')

    np.random.shuffle(graphs)
    test_length = int(len(graphs) / 10)
    train_graphs, test_graphs = graphs[:-test_length], graphs[-test_length:]
    batch_size = 32
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    print('Data loaded')

    model = GNN(3, 7, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    lossFunc = torch.nn.L1Loss(reduction='sum')

    # Load from checkpoint if available
    start_epoch = load_checkpoint(model, optimizer, 'checkpoints/big_decoder_epoch_5.pth')

    test_losses = []
    train_losses = []
    for epoch in range(start_epoch + 1, 60):
        loss = train(model, optimizer, train_loader, lossFunc)
        test_loss, lossx, lossy, lossz = test(model, test_loader, lossFunc)
        test_losses.append(test_loss)
        train_losses.append(loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr*10**7:.2f}*10^(-7)')
        print(f'Losses: x: {lossx:.4f}, y: {lossy:.4f}, z: {lossz:.4f}')
        scheduler.step()

        if epoch % 6 == 0:
            save_checkpoint(model, optimizer, epoch)
