#%%

import numpy as np
from numpy import pi
from numpy import linalg as LA
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

##################################
      #Define Basics
##################################

# single qubit basis states |0> and |1>
q0 = np.array([[1],[0]])
q1 = np.array([[0],[1]])

# Pauli Matrices
I  = np.array([[ 1, 0],[ 0, 1]])
X = np.array([[ 0, 1],[ 1, 0]])
Y = np.array([[ 0,-1j],[1j, 0]])
Z = np.array([[ 1, 0],[ 0,-1]])
HG = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])

#Creates the all zero input state.                            
def all_Zero_State(n_qubits):
        
   if n_qubits < 2:
       return 'Invalid Input : Specify at least 2 qubits'
            
   else:
       #Init State
       init_all_zero = np.kron(q0,q0)
            
       for t in range(n_qubits - 2):
           init_all_zero = np.kron(init_all_zero,q0)
                
       return init_all_zero                            

#Creates the equally superimposed product state from all zero state.                           
def equal_Superposition(n_qubits, init_all_zero):
        
    if n_qubits < 2:
        return 'Invalid Input : Specify at least 2 qubits'
            
    else:
            
        all_H = np.kron(HG,HG)
            
        for t in range(n_qubits - 2):
            all_H = np.kron(all_H,HG)
            
        equal_Superpos = all_H@init_all_zero 
                
        return equal_Superpos
    
#Analytical Ground State using Numpy's inbuilt eigh function.
def get_analytical_ground_state(H):
  e, v = LA.eigh(H)
  return np.min(e), v[:,np.argmin(e)]


#Create Unitary
def CU(Q, theta, n_qubits):
  Id = np.eye(2**n_qubits)
  if np.sum(Q@Q - Id) == 0.0:
    return np.cos(theta)*Id - 1j*np.sin(theta)*Q
  else:
    return 'Input Matrix is not involutary'


#%%