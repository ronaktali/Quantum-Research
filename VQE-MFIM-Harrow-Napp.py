
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


##################################
      #Ansatz Circuit Creation
##################################


# param_set = [[beta1,gamma1],[beta2,gamma2]] You specify params layer by layer 2 at a time for ZZs and Xs
def ansatz_vha(X_param_set, ZZ_param_set, Z_param_set, components, n_qubits, layers):

  #Initialize Ansatz to I
  ansatz = np.eye(2**n_qubits)
  
  ZZ_components = components[0]
  X_components = components[1]
  Z_components = components[2]

  for layer in range(layers):
    
    for ct1, comp1 in enumerate(ZZ_components):
      ansatz = CU(comp1, theta = ZZ_param_set[layer],n_qubits=n_qubits)@ansatz

    for ct2, comp2 in enumerate(X_components):
      ansatz = CU(comp2,  theta = X_param_set[layer],n_qubits=n_qubits)@ansatz
      
    for ct3, comp3 in enumerate(Z_components):
      ansatz = CU(comp3,  theta = Z_param_set[layer],n_qubits=n_qubits)@ansatz

  return ansatz


#Expectation Calculation - Energy
def energy_VHA(H,components, circuit_input, X_param_set, ZZ_param_set, Z_param_set, n_qubits , layers):
  psi = ansatz_vha(X_param_set = X_param_set, ZZ_param_set = ZZ_param_set, Z_param_set = Z_param_set, components = components, 
                   n_qubits = n_qubits, layers=layers)@circuit_input
  return np.real((psi.conj().T)@H@psi)[0][0]


##################################
      #Define MFIM model
##################################

#helper funnction for TFIM model creation.
def component_sums(components, n_qubits):

  ZZ_sum = np.zeros((2**n_qubits,2**n_qubits))
  X_sum = np.zeros((2**n_qubits,2**n_qubits))
  Z_sum = np.zeros((2**n_qubits,2**n_qubits))

  for zz_arr in components[0]:
    ZZ_sum += zz_arr

  for x_arr in components[1]:
    X_sum += x_arr
    
  for z_arr in components[2]:
    Z_sum += z_arr

  return ZZ_sum, X_sum, Z_sum

def array_coding_to_kron(arr, type):
  n_qubits = len(arr)
  
  if type == 'ZZ':
    convert = {0 : I, 1 : Z} #Dictionary that maps code to Pauli Matrix
    expr = np.kron(convert[arr[0]],convert[arr[1]])
    for t in range(2, n_qubits):
      expr = np.kron(expr,convert[arr[t]])

    return expr

  else:
    if type == 'X':
        convert = {0 : I, 1 : X}
        expr2 = np.kron(convert[arr[0]],convert[arr[1]])
        for k in range(2, n_qubits):
            expr2 = np.kron(expr2,convert[arr[k]])

        return expr2
    else:
        convert = {0 : I, 1 : Z}
        expr3 = np.kron(convert[arr[0]],convert[arr[1]])
        for m in range(2, n_qubits):
            expr3 = np.kron(expr3,convert[arr[m]])

        return expr3

def create_MFIM(n_qubits, g):

  if n_qubits == 2:
    return -1*np.kron(Z,Z) -g*(np.kron(X,I)+np.kron(I,X)) -g*(np.kron(Z,I)+np.kron(I,Z)), {
                                                                                           0: [np.kron(Z,Z)],
                                                                                           1: [np.kron(X,I),np.kron(I,X)], 
                                                                                           2: [np.kron(Z,I),np.kron(I,Z)]
                                                                                           }

  else:
    #This will store all the kronecker products used in Ansatz Layers
    comps = {0:[],1:[],2:[]}

    #Initializing an empty 
    mfim = np.zeros((2**n_qubits,2**n_qubits))

    # Encode ZZ Terms
    for i in range(n_qubits):
      zz_arr = np.zeros(n_qubits)
      if i < n_qubits - 1:
        zz_arr[i] = 1
        zz_arr[i+1] = 1
      else:
        zz_arr[0] = 1
        zz_arr[i] = 1

      #Call the coding function
      mfim = mfim - array_coding_to_kron(zz_arr,type='ZZ')
      #Append component
      comps[0].append(array_coding_to_kron(zz_arr,type='ZZ'))

    #X Terms
    for i in range(n_qubits):
      x_arr = np.zeros(n_qubits)
      x_arr[i] = 1

      #Call the coding function
      mfim = mfim -g* array_coding_to_kron(x_arr,type='X')
      #Append component
      comps[1].append(array_coding_to_kron(x_arr,type='X'))
      
    #Z Terms
    for i in range(n_qubits):
      z_arr = np.zeros(n_qubits)
      z_arr[i] = 1

      #Call the coding function
      mfim = mfim -g* array_coding_to_kron(z_arr,type='Z')
      #Append component
      comps[2].append(array_coding_to_kron(z_arr,type='Z'))

    return mfim, comps


##################################
      #Harrow Napp
##################################


#Helper functions to compute derivative

def all_X(X_components,param,n_qubits):
  X = np.eye(2**n_qubits)
  for component in X_components:
    X = CU(component,param,n_qubits=n_qubits)@X
  return X

def all_Z(Z_components,param,n_qubits):
  Z = np.eye(2**n_qubits)
  for component in Z_components:
    Z = CU(component,param,n_qubits=n_qubits)@Z
  return Z

def all_ZZ(ZZ_components,param,n_qubits):
  ZZ = np.eye(2**n_qubits)
  for component in ZZ_components:
    ZZ = CU(component,param,n_qubits=n_qubits)@ZZ
  return ZZ


#Gradient - Harrow Napp
def grad_harrow_napp(H, X_param_set, ZZ_param_set, Z_param_set, components, circuit_input, n_qubits,layers):

  #Prepare the common right hand side for the Harrow Napp Expression
  H_psi_right = H@ansatz_vha(X_param_set = X_param_set, ZZ_param_set = ZZ_param_set, Z_param_set = Z_param_set,
                             components = components, n_qubits = n_qubits, 
                             layers = layers)@circuit_input


  #Sum the ZZ and X components
  sum_ZZ, sum_X, sum_Z = component_sums(components, n_qubits=n_qubits) #This is implemented via a function call.
  
  #Total parameters
  param_per_layer =  3 # We always have 3 params per layer for VHA Ansatz.
  full_derivative = np.zeros(2*layers) # This is just initialization for the gradient vector

  #Derivative Expression for each param

  #ZZ params

  #Loop through all ZZ params
  for j in range(layers):
    #initialize computation for the jth ZZ derivative
    psi_left_d_ZZ = circuit_input

    #This inner loop is to loop through the circuit elements, only one of the ZZ elements will have a derivative 
    for i in range(layers):

      all_Xs = all_X(components[1],X_param_set[i],n_qubits=n_qubits)
      all_ZZs = all_ZZ(components[0],ZZ_param_set[i],n_qubits=n_qubits)
      all_Zs = all_Z(components[2],Z_param_set[i],n_qubits=n_qubits)

      if i == j:
        psi_left_d_ZZ = all_Zs@all_Xs@all_ZZs@sum_ZZ@psi_left_d_ZZ
      else:
        psi_left_d_ZZ = all_Zs@all_Xs@all_ZZs@psi_left_d_ZZ

    #Store
    full_derivative[j*param_per_layer] = -2*np.imag((psi_left_d_ZZ.conj().T)@H_psi_right)


  #X params
  for k in range(layers):
    #initialize computation for the kth X derivative
    psi_left_d_X = circuit_input

    #This inner loop is to loop through the circuit elements, only one of the X elements will have a derivative 
    for l in range(layers):
    
      all_ZZs = all_ZZ(components[0],ZZ_param_set[l],n_qubits=n_qubits)
      all_Xs = all_X(components[1],X_param_set[l],n_qubits=n_qubits)
      all_Zs = all_Z(components[2],Z_param_set[l],n_qubits=n_qubits)
  
      if l == k:
        psi_left_d_X = all_Zs@all_Xs@sum_X@all_ZZs@psi_left_d_X
      else:
        psi_left_d_X = all_Zs@all_Xs@all_ZZs@psi_left_d_X

    #Store
    full_derivative[k*param_per_layer+1] = -2*np.imag((psi_left_d_X.conj().T)@H_psi_right)
    
    
  #Z params
  for m in range(layers):
    #initialize computation for the kth X derivative
    psi_left_d_Z = circuit_input

    #This inner loop is to loop through the circuit elements, only one of the X elements will have a derivative 
    for n in range(layers):
    
      all_ZZs = all_ZZ(components[0],ZZ_param_set[n],n_qubits=n_qubits)
      all_Xs = all_X(components[1],X_param_set[n],n_qubits=n_qubits)
      all_Zs = all_Z(components[2],Z_param_set[n],n_qubits=n_qubits)
  
      if n == m:
        psi_left_d_Z = all_Zs@all_Xs@sum_Z@all_ZZs@psi_left_d_Z
      else:
        psi_left_d_Z = all_Zs@all_Xs@all_ZZs@psi_left_d_Z

    #Store
    full_derivative[k*param_per_layer+1] = -2*np.imag((psi_left_d_Z.conj().T)@H_psi_right)


  #Return all partial derivatives
  return full_derivative

##################################
      #Define Gradient Descent
##################################

#helper funnction for derivative storing and extraction.
def grad_positioning(grad):
  ZZ = []
  X = []
  Z = []
  for i in range(len(grad)):
    if i%3 == 0:
      ZZ.append(grad[i])
    else:
        if i%3 == 1:
            X.append(grad[i])
        else:
            Z.append(grad[i])
            
  return np.array(ZZ), np.array(X), np.array(Z)

#Function for Gradient Descent --> Vanilla variety.
def hn_grad_desc_quantum(H, components, X_param_set, ZZ_param_set, circuit_input, MAXITERS, eta, GRADTOL, n_qubits, layers, plotting = 'off', logging = 'off'):

  store_grad_norm = []
  store_energy = []

  #Theta is a vector ---> np.array
  theta_X = X_param_set.copy() 
  theta_ZZ = ZZ_param_set.copy()
  theta_Z = Z_param_set.copy() 

  #Keep track of number of iterations
  counter = 0 

  #Iterate
  for iter in range(MAXITERS):

    grad = grad_harrow_napp(H=H,X_param_set=theta_X,ZZ_param_set=theta_ZZ, Z_param_set=theta_Z ,components=components,
                            circuit_input=circuit_input ,n_qubits=n_qubits,layers=layers)
    
    if LA.norm(grad) < GRADTOL:
      break

    #Extract components - This is to correctly order gradient components
    ZZ, X, Z = grad_positioning(grad)

    #Update thetas
    theta_ZZ = theta_ZZ - eta*ZZ
    theta_X = theta_X - eta*X
    theta_Z = theta_Z - eta*Z
    
    #Eigenvector
    v = ansatz_vha(X_param_set = theta_X, ZZ_param_set = theta_ZZ, Z_param_set=theta_Z, components=components, n_qubits=n_qubits, 
                   layers=layers)@circuit_input

    #Eigenvalue
    e = energy_VHA(H = H ,components = components, circuit_input = circuit_input, X_param_set=theta_X, 
                   ZZ_param_set=theta_ZZ, Z_param_set=theta_Z,
                   n_qubits = n_qubits, layers = layers)

    #Keep track of number of iterations
    counter += 1
    
    #Some Periodic Logging on Terminal for large N --> if requested.
    if logging == 'on':
        #Log every 20 steps.
        if counter%20 == 0:
            print('Iteration is at ',count,' EigenValue = ',e, ' and magnitude of gradient = ',LA.norm(grad))

    #Store Gradient Norm and Energy
    store_grad_norm.append(LA.norm(grad))
    store_energy.append(e)
  
  #Some Plotting --> if requested.
  if plotting == 'on':
    plt.plot(range(counter),store_grad_norm)
    plt.title('Track Gradient Norm')
    plt.xlabel('Iteratio Number')
    plt.ylabel('L2 Norm of the Gradient')
    plt.show()

    plt.plot(range(counter),store_energy)
    plt.title('Track Cost Function')
    plt.xlabel('Iteration Number')
    plt.ylabel('Minimum Eigen Value attained')
    plt.show()

  return [theta_ZZ,theta_X, theta_Z], counter, e, v, LA.norm(grad)

#This funaction calculates the overlap of the solution with analytical ground state.
def overlap_calculator(min_gd,eigen_values, eigen_vectors):
  overlap_store = []
  for i in range(n_qubits):
    overlap_store.append(np.abs(np.vdot(min_gd,eigen_vectors[:,i]))**2)
    print('For Eigen Value ',eigen_values[i], 'overlap = ',np.abs(np.vdot(min_gd,eigen_vectors[:,i]))**2)
    print('===============================================')
    
  return None


##################################
      #Perturbative VQE
##################################

#Define perturbation step_size \eps
eps = 0.1
#Define initial Hamiltonian at g = 0

#For total steps = 0 --> 10, go from g = 0 to g = 1
    #Loop 10 times, each time restarting the problem with outputs from previous step.
        #Minitor convergence
        #Monitor performance - time/#iterations


##################################
      # 3 Run Examples below.
##################################

#%%

#RUN FOR N=4 and g = 0.5

n_qubits = 4
g = 0.5
layers = 2

X_param_set = np.array([pi/3,pi/3]) 
ZZ_param_set = np.array([pi/3,pi/3])
Z_param_set = np.array([pi/3,pi/3]) 

circuit_input = equal_Superposition(n_qubits,all_Zero_State(n_qubits))

#Create the TFIM Model
H, components = create_MFIM(n_qubits = n_qubits, g = g)

#Diagonalize
e, v = LA.eigh(H)

#Print min eigen value
print('Min Eigen = ', np.min(e))

#Call Gradient Descent
h_theta, count, h_eigen_value, h_eigen_vector, h_grad = hn_grad_desc_quantum(H = H, components = components, 
                                                                          X_param_set = X_param_set, ZZ_param_set = ZZ_param_set, 
                                                                          Z_param_set = Z_param_set,
                                                                          circuit_input=circuit_input,
                                                                          MAXITERS=1000, eta = 0.001,  GRADTOL = 0.00001,
                                                                          n_qubits = n_qubits, layers = layers, plotting = 'on', logging='on')

#Print min eigen value
print('Caluclated Min Eigen = ', h_eigen_value)

#Print Error
print('Absolute Error in min eigen value calculation = ', np.abs(h_eigen_value - np.min(e)))

overlap_calculator(min_gd = h_eigen_vector, eigen_values = e, eigen_vectors = v)

#%%
#RUN FOR N=4 and g = 1.1

n_qubits = 4
layers = 2

X_param_set = np.array([pi/8,pi/8]) 
ZZ_param_set = np.array([pi/8,pi/8])
Z_param_set = np.array([pi/8,pi/8])


circuit_input = equal_Superposition(n_qubits,all_Zero_State(n_qubits))

#Create the TFIM Model
H1, components1 = create_MFIM(n_qubits = 4, g = 1.1)

#Diagonalize
e, v = LA.eigh(H1)

#Print min eigen value
print('Min Eigen = ', np.min(e))

#Call Gradient Descent
h_theta, count, h_eigen_value, h_eigen_vector, h_grad = hn_grad_desc_quantum(H = H1, components = components1, 
                                                                          X_param_set = X_param_set, ZZ_param_set = ZZ_param_set, 
                                                                          Z_param_set = Z_param_set,
                                                                          circuit_input=circuit_input,
                                                                          MAXITERS=1000, eta = 0.001,  GRADTOL = 0.00001,
                                                                          n_qubits = n_qubits, layers = layers, plotting = 'on',logging='on')

#Print min eigen value
print('Caluclated Min Eigen = ', h_eigen_value)

#Print Error
print('Absolute Error in min eigen value calculation = ', np.abs(h_eigen_value - np.min(e)))

#Calculate Overlap
overlap_calculator(min_gd = h_eigen_vector, eigen_values = e, eigen_vectors = v)

#%%
#RUN FOR N=8 and g = 1.1

n_qubits = 8
layers = 4 #Usually N/2 but for large N, it may have to be less.
g = 1.1

X_param_set = np.array([pi/8,pi/8,pi/8,pi/8]) 
ZZ_param_set = np.array([pi/8,pi/8,pi/8,pi/8])
Z_param_set = np.array([pi/8,pi/8,pi/8,pi/8])

circuit_input = equal_Superposition(n_qubits,all_Zero_State(n_qubits))

#Create the TFIM Model
H1, components1 = create_MFIM(n_qubits = n_qubits, g = g)

#Diagonalize
e, v = LA.eigh(H1)

#Print min eigen value
print('Min Eigen = ', np.min(e))


#Call Gradient Descent
h_theta, count, h_eigen_value, h_eigen_vector, h_grad = hn_grad_desc_quantum(H = H1, components = components1, 
                                                                          X_param_set = X_param_set, ZZ_param_set = ZZ_param_set, 
                                                                          Z_param_set = Z_param_set,
                                                                          circuit_input=circuit_input,
                                                                          MAXITERS=100, eta = 0.01,  GRADTOL = 0.01,
                                                                          n_qubits = n_qubits, layers = layers, plotting = 'on',logging = 'on')

#Print min eigen value
print('Calculated Min Eigen = ', h_eigen_value)

#Print Error
print('Absolute Error in min eigen value calculation = ', np.abs(h_eigen_value - np.min(e)))

#Calculate Overlap
overlap_calculator(min_gd = h_eigen_vector, eigen_values = e, eigen_vectors = v)


##################################
      #End of Program
##################################
