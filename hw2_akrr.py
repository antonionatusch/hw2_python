import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def rbf_kernel_matrix(A, B, gamma):
    """
    Rectangular RBF kernel K(A,B) with entries exp(-gamma ||a - b||^2).
    A: (n_a, p), B: (n_b, p)
    returns: (n_a, n_b)
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    A_sq = np.sum(A**2, axis=1, keepdims=True)          # (n_a, 1)
    B_sq = np.sum(B**2, axis=1, keepdims=True).T        # (1, n_b)
    cross = A @ B.T                                     # (n_a, n_b)
    d2 = A_sq + B_sq - 2.0 * cross
    return np.exp(-gamma * d2)

data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)


num_train = int(0.75*n)
num_test = int(0.25*n)


sample_train = data[0:num_train,0:-1]
sample_test = data[n-num_test:,0:-1]
label_train = data[0:num_train,-1]
label_test = data[n-num_test:,-1]

# pick lamda for KRR/AKRR 
lamda = 0.5

# pick gamma for RBF kernel (in both KRR/AKRR) 
gamma = 0.05

# baseline KRR from scikit 
model = KernelRidge(alpha=lamda, kernel = 'rbf', gamma = gamma)
model.fit(sample_train,label_train)
label_test_pred = model.predict(sample_test)
er_krr = mean_squared_error(label_test, label_test_pred)

# now pick at least five values for m
m_values = [10, 20, 40, 80, 120, 200, 300, 350, 800]
m_values = [m for m in m_values if m <= num_train] # to using m larger than training number

er_base = []
er_yours = []

for m in m_values:
    
    # baseline performance, fixed across m
    er_base.append(er_krr) 
    
    basis_X = sample_train[:m, :]
    
    # now, implement AKRR by yourself 
    
    # rect kernels
    K_train = rbf_kernel_matrix(sample_train, basis_X, gamma)
    K_test = rbf_kernel_matrix(sample_test, basis_X, gamma)
    
    # basis-basis kernel
    
    K_m = rbf_kernel_matrix(basis_X, basis_X, gamma)
    
    # solving equation found in Part 1, Task 1
    
    A = K_train.T @ K_train + lamda * K_m
    b = K_train.T @ label_train
    
    # Solving by Cholesky
    
    L = np.linalg.cholesky(A)
    y_mid = np.linalg.solve(L, b)
    alpha_tilde = np.linalg.solve(L.T, y_mid)
    
    # Do predictions on test set
    
    label_test_pred_akrr = K_test @ alpha_tilde
    
    
    # store your MSE
    er_yours.append(mean_squared_error(label_test, label_test_pred_akrr))
    

# the following code will plot Figure 1. 
plt.figure()
plt.plot(m_values,er_base, '--o', label='Scikit Error')
plt.plot(m_values,er_yours, '--s', label='AKRR (first m basis) Error')
plt.xlabel('m')
plt.ylabel('MSE')
plt.legend()
plt.show()

