import utils
import numpy as np 
import torch

model = utils.Model()

soap_H_plane_test = np.load("../H-Plane_for-test/H-plane_soap.npy")
soap_H_plane_test.shape
