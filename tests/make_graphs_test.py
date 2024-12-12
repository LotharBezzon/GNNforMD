import unittest
import torch
from torch_geometric.data import Data
from DataProcessing import make_graphs

class TestMakeGraphs(unittest.TestCase):

    def setUp(self):
        self.data = [
            {
                'timestep': 0,
                'num_atoms': 2,
                'box_size': torch.tensor([10.0, 10.0, 10.0], dtype=torch.float),
                'atoms': [
                    {'id': 0, 'mol': 1, 'type': 0, 'x': 1.0, 'y': 1.0, 'z': 1.0, 'fx': 0.1, 'fy': 0.2, 'fz': 0.3},
                    {'id': 1, 'mol': 1, 'type': 1, 'x': 2.0, 'y': 2.0, 'z': 2.0, 'fx': 0.4, 'fy': 0.5, 'fz': 0.6}
                ]
            }
        ]
        self.charges = {0: 1.0, 1: -1.0}
        self.LJ_params = {0: (0.1, 0.2), 1: (0.3, 0.4)}
        self.cutoff = 2.3

    def test_output_type(self):
        graphs = make_graphs(self.data, self.charges, self.LJ_params, self.cutoff)
        self.assertIsInstance(graphs, list)
        self.assertTrue(all(isinstance(graph, Data) for graph in graphs))

    def test_number_of_graphs(self):
        graphs = make_graphs(self.data, self.charges, self.LJ_params, self.cutoff)
        self.assertEqual(len(graphs), len(self.data))

    def test_graph_structure(self):
        graphs = make_graphs(self.data, self.charges, self.LJ_params, self.cutoff)
        graph = graphs[0]
        self.assertEqual(graph.x.shape, (2, 3))
        self.assertEqual(graph.edge_index.shape[0], 2)
        self.assertEqual(graph.edge_attr.shape[1], 7)
        self.assertEqual(graph.y.shape, (2, 3))

    def test_edge_attributes(self):
        graphs = make_graphs(self.data, self.charges, self.LJ_params, self.cutoff)
        graph = graphs[0]
        edge_attr = graph.edge_attr
        self.assertTrue(torch.all(edge_attr[:, 3] < self.cutoff))

    def test_node_features(self):
        graphs = make_graphs(self.data, self.charges, self.LJ_params, self.cutoff)
        graph = graphs[0]
        expected_features = torch.tensor([[1.0, 0.1, 0.2], [-1.0, 0.3, 0.4]], dtype=torch.float)
        self.assertTrue(torch.equal(graph.x, expected_features))

if __name__ == '__main__':
    unittest.main()