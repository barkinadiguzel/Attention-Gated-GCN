from chemistry.graph_builder import mol_to_graph
from gcn.gcn_core import AttentionGatedGCN
from rdkit import Chem

# (2-propanol)
mol = Chem.MolFromSmiles("CCO")
X, A = mol_to_graph(mol)

model = AttentionGatedGCN(in_dim=X.size(1), hidden_dim=128, num_layers=3, num_heads=4)
y_pred, node_features, graph_feature = model(X, A)

print("Predicted property:", y_pred)
print("Node features shape:", node_features.shape)
print("Graph feature shape:", graph_feature.shape)
