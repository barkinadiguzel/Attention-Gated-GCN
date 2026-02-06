from rdkit import Chem
import torch
from .atom_features import get_atom_features

def mol_to_graph(mol):
    N = mol.GetNumAtoms()
    X = torch.stack([get_atom_features(a) for a in mol.GetAtoms()])

    A = torch.zeros((N,N), dtype=torch.float)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        A[i,j] = 1
        A[j,i] = 1
    A += torch.eye(N)  # self-loop
    return X, A
