import torch

def get_atom_features(atom):
    atom_types = ['H','C','N','O','F','P','S','Cl','Br','I']
    atom_type_feat = [int(atom.GetSymbol() == t) for t in atom_types]

    num_h = [atom.GetTotalNumHs()]
    valence = [atom.GetTotalValence()]
    aromatic = [int(atom.GetIsAromatic())]

    features = atom_type_feat + num_h + valence + aromatic
    return torch.tensor(features, dtype=torch.float)
