import torch
from torch.nn import Sequential, Linear, Dropout, ModuleList, PReLU
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
        normalize (bool, optional): Whether to apply batch normalization. Default is False.
        bias (bool, optional): Whether to include bias in the linear layers. Default is False.

    Attributes:
        mlp (torch.nn.Sequential): The sequential container of the MLP layers.
    """
    def __init__(self, in_channels, 
                 out_channel, 
                 hidden_dim=128, 
                 hidden_num=3, 
                 normalize=False, 
                 bias=False):
        super().__init__()
        self.layers = [Linear(in_channels, hidden_dim), PReLU()]
        for _ in range(hidden_num):
            self.layers.append(Dropout(0.1))
            self.layers.append(Linear(hidden_dim, hidden_dim, bias=bias))
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
        \mathbf{x}_i^{\prime} = \text{MLP}\left(
        \Sigma_{j \in \mathcal{N}(i)} \,\{\text{MLP}
        \left((\mathbf{x}_i \, *\, \mathbf{x}_j)\, ||\, \mathbf{e}_{ji}\right)\} \right)

    Args:
        in_channels (int): Number of input features for each node.
        out_channels (int): Number of output features for each node.

    Attributes:
        mlp (mlp): A multi-layer perceptron (MLP) used to process messages.
        mlp_out (mlp): A multi-layer perceptron (MLP) used to process the aggregated messages.

    Methods:
        forward(edge_index, v, e): Performs the message passing and aggregation.
        message(v_i, v_j, e): Constructs messages from node features and edge features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum')
        self.mlp = mlp(2*in_channels, out_channels, hidden_num=2)
        self.mlp_out = mlp(out_channels, out_channels, hidden_num=2)

    def forward(self, edge_index, v,  e):
        accumulated_message= self.propagate(edge_index, v=v, e=e)
        return self.mlp_out(accumulated_message)

    def message(self, v_i, v_j, e):
        return self.mlp(torch.cat([v_i * v_j, e], dim=-1))

class GNN(torch.nn.Module):
    """
    A graph neural network (GNN) model.

    Args:
        node_dim (int): Number of input features for each node.
        edge_dim (int): Number of input features for each edge.
        out_dim (int): Number of output features.
        embedding_dim (int, optional): Dimension of the embeddings for nodes and edges. Default is 128.
        mp_num (int, optional): Number of message passing layers. Default is 3.

    Attributes:
        node_encoder (mlp): MLP to encode node features.
        edge_encoder (mlp): MLP to encode edge features for the first message passing layer.
        far_edge_encoder (mlp): MLP to encode edge features for tother message passing layers.
        message_passing_layers (ModuleList): List of message passing layers and normalization layers.
        decoder (mlp): MLP to decode the final node embeddings to output features.

    Methods:
        forward(data): Forward pass of the GNN model.
        Args:
            data (torch_geomatric.data.Data): Input graph.
    """
    def __init__(self, 
                 node_dim, 
                 edge_dim, 
                 out_dim, 
                 embedding_dim=128, 
                 mp_num=3):
        super().__init__()
        torch.manual_seed(12345)
        self.node_encoder = mlp(node_dim, embedding_dim, hidden_num=2)
        self.edge_encoder = mlp(edge_dim, embedding_dim, hidden_num=2)
        self.far_edge_encoder = mlp(edge_dim-3, embedding_dim, hidden_num=2)   # Does not include bond type
        self.message_passing_layers = ModuleList()
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
                    v = v + layer(data.edge_index, v, far_e)    # Residual connection
            else:
                v = layer(v)

        return self.decoder(v)

class equivariantMPLayer(MessagePassing):
    """
    A message passing layer for an equivariant graph neural network (GNN).

    Args:
        in_channels (int): Number of input features for each node.
        first (bool, optional): Whether this is the first message passing layer. Default is False.

    Attributes:
        mlp1 (mlp): A multi-layer perceptron (MLP) used in the first message passing layer.
        mlp2 (mlp): A multi-layer perceptron (MLP) used in subsequent message passing layers.
        first (bool): Indicates whether this is the first message passing layer.

    Methods:
        forward(edge_index, v, e, direction, f=None): Performs the message passing and aggregation.
        message(v_i, v_j, e, direction, f_j): Constructs messages from node features and edge features.
    """
    def __init__(self, in_channels, first=False):
        super().__init__(aggr='sum')
        self.mlp1 = mlp(2*in_channels, 1, hidden_num=6, bias=False, hidden_dim=96)
        self.mlp2 = mlp(4*in_channels, 4, hidden_num=6, bias=True, hidden_dim=96)
        self.first = first
        
    def forward(self, edge_index, v, e, direction, f=None):
        f = self.propagate(edge_index, v=v, e=e, f=f, direction=direction)
        return f
    
    def message(self, v_i, v_j, e, direction, f_j):
        if self.first:
            temp = self.mlp1(torch.cat([v_i * v_j, e], dim=-1))    # temp is the force modulus
            return torch.cat([temp * direction, temp], dim=-1)     # return the force vector and the modulus (to learn surrounding atoms)
        else:
            temp = self.mlp2(torch.cat([v_i, v_j, e, f_j], dim=-1))
            return temp
        

class equivariantGNN(torch.nn.Module):
    """
    An equivariant graph neural network (GNN) model.

    Args:
        node_dim (int): Number of input features for each node.
        edge_dim (int): Number of input features for each edge.
        embedding_dim (int): Dimension of the embeddings for nodes and edges.
        mp_num (int): Number of message passing layers.
        encoders_hidden_num (int): Number of hidden layers in the node and edge encoders.

    Attributes:
        node_encoder (mlp): MLP to encode node features.
        edge_encoder (mlp): MLP to encode edge features.
        message_passing_layers (ModuleList): List of message passing layers.
        decoder (mlp): MLP to decode the final node embeddings to output features.

    Methods:
        forward(data): Forward pass of the GNN model.
    """
    def __init__(self, 
                 node_dim=3, 
                 edge_dim=4, 
                 embedding_dim=32, 
                 mp_num=3,
                 encoders_hidden_num=3):
        super().__init__()
        self.node_encoder = mlp(node_dim, embedding_dim, hidden_num=encoders_hidden_num, hidden_dim=embedding_dim)
        #self.far_node_encoder = mlp(node_dim, embedding_dim, hidden_num=encoders_hidden_num, hidden_dim=embedding_dim)
        self.edge_encoder = mlp(edge_dim, embedding_dim, hidden_num=encoders_hidden_num, hidden_dim=embedding_dim)
        self.far_edge_encoder = mlp(edge_dim, embedding_dim, hidden_num=encoders_hidden_num, hidden_dim=embedding_dim)
        self.force_encoder = mlp(4, embedding_dim, hidden_num=encoders_hidden_num, hidden_dim=embedding_dim)
        self.message_passing_layers = ModuleList([GraphNorm(embedding_dim), equivariantMPLayer(embedding_dim, first=True)])
        for _ in range(mp_num-1):
            self.message_passing_layers.append(GraphNorm(embedding_dim))
            self.message_passing_layers.append(equivariantMPLayer(embedding_dim))
                
    def forward(self, data):
        v = self.node_encoder(data.x)
        #far_v = self.far_node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr[:,:4])
        distance = self.far_edge_encoder(data.edge_attr[:,3:])    # the second index must be like this to have the correct shape
        direction = data.edge_attr[:,4:]
        
        first = True
        for layer in self.message_passing_layers:
            if isinstance(layer, equivariantMPLayer):
                if first:
                    f = layer(data.edge_index, v, e, direction)
                    embedded_f = self.force_encoder(f)
                    first = False
                else:
                    f = f + layer(data.edge_index, v, distance, direction, embedded_f)
            else:
                if first:
                    v = layer(v)
                else:
                    embedded_f = layer(embedded_f)
        return f[:,:3]

class GATModel(torch.nn.Module):
    """
    A graph attention network (GAT) model. Didn't show great performance and is still quite raw.
    """
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