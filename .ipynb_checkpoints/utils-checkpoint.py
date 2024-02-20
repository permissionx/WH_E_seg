import torch
from torch import nn
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np 


device = "cuda" if torch.cuda.is_available() else "cpu"
# SOAP descriptors
def Descriptors(atoms, positions):
	"""
    Generates and scales the SOAP descriptors for a given set of atoms and their positions.

    Parameters:
    - atoms (list): all W atoms by loaded by ase (Atomic Simulation Environment) 
    - positions (array): Array of TIS positions. positions.shape = (position_number, 3)
    """
   
	species = ["W"]
	rcut = 6 
	nmax = 8
	lmax = 6
	sigma = 0.3
	# size = 252
	soap = SOAP(
	    species=species,
	    periodic=True,
	    rcut=rcut,
	    nmax=nmax,
	    lmax=lmax,
	    sigma = sigma
	)

	# Scaler
	descriptors_soap = soap.create(atoms, positions=positions)
	scaler = pickle.load(open("scaler.pkl", 'rb'))
	descriptors_soap_scaled = scaler.transform(descriptors_soap)
	soaps = torch.Tensor(descriptors_soap_scaled).to(device)
	return soaps


# ANN model
class NeuralNetwork(nn.Module):
    def __init__(self, soap_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_rule_stack = nn.Sequential(
            nn.Linear(soap_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.linear_rule_stack.apply(self.init_weights)
    
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self,x):
        x = nn.functional.normalize(x)
        y = self.linear_rule_stack(x)
        return y



def Model(soap_size=252):
	model = NeuralNetwork(soap_size).to(device)
	model_state_dict = torch.load('tis.model')
	model.load_state_dict(model_state_dict)
	return model






