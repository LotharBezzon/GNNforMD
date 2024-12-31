import torch
from torch.nn import Sequential, Linear, GELU, Dropout, ReLU, ModuleList, PReLU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, BatchNorm, LayerNorm, GraphNorm

class mlp(torch.nn.Module):
    """
    A multi-layer perceptron (MLP) neural network module.

    Args:
        in_channels (int): Number of input features.
        out_channel (int): Number of output features.
        hidden_dim (int, optional): Number of hidden units in each hidden layer. Default is 128.
        hidden_num (int, optional): Number of hidden layers. Default is 3.
        activation (torch.nn.Module, optional): Activation function to use. Default is GELU().

    Attributes:
        mlp (torch.nn.Sequential): The sequential container of the MLP layers.
    """
    def __init__(self, in_channels, out_channel, hidden_dim=128, hidden_num=3, normalize=False):
        super().__init__()
        self.layers = [Linear(in_channels, hidden_dim), PReLU()]
        for _ in range(hidden_num):
            self.layers.append(Dropout(0.1))
            self.layers.append(Linear(hidden_dim, hidden_dim, bias=False))
            if normalize:
                self.layers.append(BatchNorm(in_channels))
            self.layers.append(PReLU())
        self.layers.append(Linear(hidden_dim, out_channel))
        self.mlp = Sequential(*self.layers)
        self._init_parameters()

    def _init_parameters(self):
         for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
            return self.mlp(x)

class MPLayer(MessagePassing):
    """
    A message passing layer for a graph neural network (GNN).
    .. math::
        \mathbf{x}_i^{\prime} = \mathbf{x}_i + \left(
        \text{mean}_{j \in \mathcal{N}(i)} \,\{\text{MLP}
        \left((\mathbf{x}_i \, +\, \mathbf{x}_j)\, ||\, \mathbf{e}_{j,i}\right)\} \right)

    Args:
        in_channels (int): Number of input features for each node.
        out_channels (int): Number of output features for each node.

    Attributes:
        mlp (mlp): A multi-layer perceptron (MLP) used to process messages.

    Methods:
        forward(edge_index, v, e): Performs the message passing and aggregation.
        message(v_i, v_j, e): Constructs messages from node features and edge features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum')
        self.mlp = mlp(2*in_channels, out_channels)

    def forward(self, edge_index, v,  e):
        accumulated_message= self.propagate(edge_index, v=v, e=e)
        return accumulated_message

    def message(self, v_i, v_j, e):
        return self.mlp(torch.cat([v_i * v_j, e], dim=-1))
        #return self.mlp(v_i + v_j + e)

class GNN(torch.nn.Module):
    """
    A graph neural network (GNN) model with message passing layers.

    Args:
        node_dim (int): Number of input features for each node.
        edge_dim (int): Number of input features for each edge.
        embedding_dim (int): Dimension of the embeddings for nodes and edges.
        out_dim (int): Number of output features.
        mp_num (int): Number of message passing layers.

    Attributes:
        node_encoder (mlp): MLP to encode node features.
        edge_encoder (mlp): MLP to encode edge features.
        message_passing_layers (ModuleList): List of message passing layers and normalization layers.
        norm_layer (BatchNorm1d): Batch normalization layer for edge features.
        decoder (mlp): MLP to decode the final node embeddings to output features.

    Methods:
        forward(data): Forward pass of the GNN model.
        Args:
            data (torch_geomatric.data.Data): Input graph.
    """
    def __init__(self, node_dim, edge_dim, out_dim, embedding_dim=128, mp_num=3):
        super().__init__()
        torch.manual_seed(12345)
        self.node_encoder = mlp(node_dim, embedding_dim, hidden_num=2)
        self.edge_encoder = mlp(edge_dim, embedding_dim, hidden_num=2)
        self.far_edge_encoder = mlp(edge_dim-3, embedding_dim, hidden_num=2)
        self.message_passing_layers = ModuleList()
        self.norm_layer = GraphNorm(embedding_dim)
        for _ in range(mp_num):
            self.message_passing_layers.append(GraphNorm(embedding_dim))
            self.message_passing_layers.append(MPLayer(embedding_dim, embedding_dim))
        self.decoder = mlp(embedding_dim, out_dim, hidden_num=2, normalize=False)
        
        
    def forward(self, data):
        v = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        far_e = self.far_edge_encoder(data.edge_attr[:,3:])
        
        first = True
        for layer in self.message_passing_layers:
            if isinstance(layer, MPLayer):
                if first:
                    v = layer(data.edge_index, v, e)
                    first = False
                else:
                    v = v + layer(data.edge_index, v, far_e)
            else:
                v = layer(v)

        return self.decoder(v)

class equivariantMPLayer(MessagePassing):
    def __init__(self, first=False):
        super().__init__(aggr='sum')
        self.mlp1 = mlp(7, 1)
        self.mlp2 = mlp(11, 1)
        self.first = first
        
    def forward(self, edge_index, v, e, direction, f=None):
        f = self.propagate(edge_index, v=v, e=e, f=f, direction=direction)
        return f
    
    def message(self, v_i, v_j, e, direction, f_j):
        if self.first:
            temp = self.mlp1(torch.cat([v_i * v_j, e], dim=-1))
            return torch.cat([temp * direction, temp], dim=-1)
        else:
            temp = self.mlp2(torch.cat([v_i, v_j, e, f_j], dim=-1))
            return torch.cat([temp * direction, temp], dim=-1)
        

class equivariantGNN(torch.nn.Module):
    def __init__(self, embedding_dim=128, mp_num=3):
        super().__init__()
        self.message_passing_layers = ModuleList([GraphNorm(embedding_dim), equivariantMPLayer(first=True)])
        for _ in range(mp_num-1):
            self.message_passing_layers.append(GraphNorm(embedding_dim))
            self.message_passing_layers.append(equivariantMPLayer())
                
    def forward(self, data):
        v = data.x
        e = data.edge_attr[:,:4]
        distance = data.edge_attr[:,3:4]    # the second index must be writed like this to have the correct shape
        direction = data.edge_attr[:,4:]
        
        first = True
        for layer in self.message_passing_layers:
            if isinstance(layer, equivariantMPLayer):
                if first:
                    f = layer(data.edge_index, v, e, direction)
                    first = False
                else:
                    f = f + layer(data.edge_index, v, distance, direction, f)
            else:
                pass
                #v = layer(v)
        return f[:,:3]

class GATModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, embedding_dim=32, num_layers=4, heads=8):
        super(GATModel, self).__init__()
        self.node_encoder = mlp(node_dim, embedding_dim, hidden_dim=embedding_dim, hidden_num=1)
        self.edge_encoder = mlp(edge_dim, embedding_dim, hidden_dim=embedding_dim, hidden_num=1)
        self.message_passing_layers = ModuleList([LayerNorm(embedding_dim), GATConv(embedding_dim, embedding_dim, edge_dim=embedding_dim, heads=heads, residual=True)])
        for _ in range(num_layers - 1):
            self.message_passing_layers.append(LayerNorm(embedding_dim * heads))
            self.message_passing_layers.append(GATConv(embedding_dim * heads, embedding_dim, edge_dim=embedding_dim, heads=heads, residual=True))
            self.dropout = Dropout(0.1)
        self.decoder = mlp(embedding_dim * heads, out_dim, hidden_dim=embedding_dim, hidden_num=1)
        

    def forward(self, data):
        v = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        
        for layer in self.message_passing_layers:
            if isinstance(layer, GATConv):
                v = layer(v, data.edge_index, edge_attr=e)
            else:
                v = layer(v)
        return self.decoder(v)