import numpy as np
from sklearn.decomposition import nmf, nmf_new
from sklearn.preprocessing import normalize
from numpy import linalg
from util import *
import pdb

def non_negative_factorization(X, s0, tol=1e-4, max_iter=200):
    d = X.shape[0]
    U,V = init(d,s0)
    iter_ = 0
    #F = np.zer
    while True:
        while (iter_>30 or (F[-31:-2].mean()-F[-1])/F[-1]> 0.001) and iter_<max_iter:
            iter_ += 1
    return U,V,n_iter

def subroutine(X,U,V,max_iter = 200, tol = 0.001):
    iter_ = 0
    F =np.zeros(max_iter)
    while (iter_<30 or (F[iter_-30:iter_-1].mean()-F[iter_])/F[iter_]> tol) and iter_<max_iter:
        iter_ += 1
        U,V,s = reduce(U,V)
        U,V = palm(X,U,V)
        F[iter_] = linalg.norm(X - U@V.T)
        print(f'Complete iter {iter_}: {F[iter_]}')
    return U,V

def palm(X, U, V, lambda_ = 1e-06, eps = 1e-14, gamma1 = 1.1, gamma2 = 1.2 ):
    c = 1/(gamma1 * (linalg.norm(V.T@V)+ lambda_/eps))
    d = 1/(gamma2 * (linalg.norm(U.T@U) + lambda_ * np.sqrt(U.shape[0]) * linalg.norm(U)))
    F_U = -(X - U@V.T)@V + lambda_ * U @ np.diag([linalg.norm(V[:,j])/linalg.norm(U[:,j]) for j in range(U.shape[1])])
    U = projection(U - c * F_U)
    F_V = -(X - U@V.T).T@U + lambda_ * V @ np.diag([linalg.norm(U[:,j])/linalg.norm(V[:,j]) for j in range(U.shape[1])])
    V = projection(V - d * F_V)
    return U,V

def reduce(U,V, drop_threshold = 0.001):
    norm = linalg.norm(U, ord = 2, axis = 0)
    drop = np.where(norm<drop_threshold)
    U = np.delete(U, drop, axis = 1)
    V = np.delete(V, drop, axis = 1)
    return U,V,U.shape[1] - len(drop)

def compress1(U, V):
    import sympy
    d,s = U.shape
    combination = [[U[j,i] * V[k,i] for i in range(s)] for j in range(d) for k in range(d)]
    _, columns = sympy.Matrix(t).rref(combination)
    U = normalize(U[:,columns], axis = 1, norm = 'l1')
    V = V[:,columns]
    return U,V
    
def compress2(U, V, ind_threshold):
    pass
      
def expand(U,V,u,v):
    U = np.diag(np.ones(U.shape[0]) - kappa * u)@U
    U = np.concatenate((U, kappa * u), axis = 1)
    V = np.concatenate((V, 1/(v.T@np.ones(V.shape[0]))@v), axis = 1)
    return U,V

def positive_normalize(x):
    x = x * (x>0)
    return normalize(x)
    
def check_global_optimality(P,X,mu, threshold, iter_, lr,extra_eps, lambda_):
    t = mu@np.ones(X.shape[0]) - 2(X - P)
    u,v = init(X.shape[0],1)
    change = 1
    iter_ = 0
    new_sigma = u.T@t@v
    while change>threshold and iter_<100:
        old_sigma = new_sigma
        u = positive_normalize(u + lr * t@v)
        v = positive_normalize(v + lr * t.T@u)
        new_sigma = u.T@t@v
        change = new_sigma / old_sigma - 1
    return new_sigma < (1+extra_eps) * lambda_, u, v

def norm(X):
    return linalg.norm(X, ord = 2, axis = 0)

def projection(X):
    d,s = X.shape
    proj = np.zeros([d,s])
    for i in range(d):
        l = np.argmax([sum([X[i,k] - X[i,j] for k in range(j)]) for j in range(s)])
        eta = 1/l * (1 - sum(X[i,:l]))
        proj[i,:] = X[i,:] + eta * np.ones(s)
        proj[i,:] = proj[i,:] * (proj[i,:]>0)
    return proj

def init(d, s0):
    U = normalize(np.random.rand(d,s0), axis = 1, norm = 'l1')
    V = normalize(np.random.rand(d,s0), axis = 0, norm = 'l1')
    return U,V

if __name__ == "__main__":
    U0,V0 = init(2000,5)
    P = U0 @ V0.T
    U, V = init(2000,300)
    subroutine(P,U,V) 