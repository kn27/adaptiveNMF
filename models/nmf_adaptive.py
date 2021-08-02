import numpy as np
#from sklearn.decomposition import nmf
from sklearn.preprocessing import normalize
from numpy import linalg
from util import *
import pdb
import logging
from scipy.linalg import svd 
import hickle
    
log = logging.getLogger('runner.optimizer')
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('[%(levelname)7s] - %(message)s')
# handler.setFormatter(formatter)
# log.addHandler(handler)

# fh = logging.FileHandler(f'runner.log')
# fh.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# log.addHandler(fh)

def overall_check(E, P, U, V, lambda_):
    return {'overall': linalg.norm(P - U@V.T), 'F': {1/2*linalg.norm(E@(P-U@V.T))**2}, 'regularize':lambda_*sum([linalg.norm(U[:,i]) * linalg.norm(V[:,i]) for i in range(U.shape[1])])}

def cost(E,X,U,V,lambda_):
    d,s = U.shape
    return 1/2*linalg.norm(E@(X-U@V.T))**2 + lambda_*sum([linalg.norm(U[:,i]) * linalg.norm(V[:,i]) for i in range(s)])


def non_negative_factorization(P, s0,starting_p = 1, starting_kappa = 0.5 , kappa_threshold = 1e-5, lr = 0.1, drop_threshold = 1e-8, max_iter = 10, inner_max_iter = 200,  lambda_ = 1e-06, eps = 1e-4, gamma1 = 1.1, gamma2 = 1.2, capture:bool = True, outputpath:str = './', local = False):
    #pdb.set_trace()
    d = P.shape[0]
    #E = np.diag(np.sum(P, axis = 0))/d
    E = np.diag(np.ones(d))/d
    U,V = init(d,s0)
    iter_ = 1 
    if local:
        global log
        #del log
        log = logging.getLogger('test')
        log.setLevel(logging.INFO)
        #handler = logging.StreamHandler(sys.stdout)
        #handler.setLevel(logging.INFO)
        #formatter = logging.Formatter('[%(levelname)7s] - %(message)s')
        #handler.setFormatter(formatter)
        #log.addHandler(handler) 
    while True:
        #U,V = compress(U,V)
        log.info(f'MAIN - Start subroutine')
        U,V,mu,F = subroutine(E,P,U,V, eps = eps, drop_threshold = drop_threshold, max_iter = inner_max_iter, lambda_ = lambda_,  gamma1 = gamma1, gamma2 = gamma2)
        global_check, u, v = check_global_optimality(E, P, U@V.T, mu,lambda_ = lambda_, max_iter= 200, lr = lr)
        if capture:
            hickle.dump((U,V),f'{outputpath}Output_r={U.shape[1]:02d}')
        if global_check:
            log.info(f'MAIN - Achieve global optimum!')
            break
        elif iter_ == max_iter:
            log.info(f'MAIN - Exit at iter = {iter_}')
            break
        else:
            #NOTE:search for best kappa
            p = starting_p
            while True:
                #kappa = 0.5**p/np.asscalar(max(abs(u)))
                #pdb.set_trace()
                kappa = starting_kappa**p/u
                from numpy import inf
                kappa[kappa == inf] = 0
                kappa[kappa == -inf] = 0
                kappa = np.diag(kappa.flatten())
                U_test,V_test = expand(U,V,u,v,kappa)
                newcost = cost(E,P,U_test,V_test,lambda_)
                #log.debug(f'MAIN - Test p = {p}, kappa = {kappa}, cost = {newcost}') 
                if newcost < F[-1]*(1-kappa_threshold):
                    log.info(f'MAIN - Choose kappa = {kappa}, p = {p}, cost = {newcost}, sum_by_column = {np.sum(U_test, axis = 0)}')
                    log.info(f'MAIN - Number of columns = {U_test.shape[1]}')
                    U,V = U_test, V_test
                    break
                elif (np.max(kappa.flatten()) < 1e-8).all():
                    log.info(f'MAIN - Achieve global optimum by kappa!')# = {kappa}!')
                    return U,V,mu,E
                else:
                    p += 1                    
            iter_ += 1
    return U,V,mu,E

def validate(X):
    return all(np.isclose(X@np.ones(X.shape[1]),np.ones(X.shape[0]))) & all(X.flatten() >= 0)

def subroutine(E,X,U,V,drop_threshold = 1e-8, max_iter = 40, lambda_ = 1e-06, eps = 1e-12, gamma1 = 1.1, gamma2 = 1.2 ):
    d,s = U.shape
    F = []
    #log.info(f'OPTIMIZER - Starting error {cost(E,X,U,V,lambda_)}')
    iter_ = 0
    #log.debug(f'min norm U {min(norm_by_axis(U))} and min norm V {min(norm_by_axis(V.T))}')
    while True:
        U,V,s = reduce(U,V,drop_threshold)
        U,V = palm(E,X,U,V,lambda_, eps, gamma1, gamma2)
        F.append(cost(E,X,U,V,lambda_))
        #norm_U = norm_by_axis(U)
        #norm_V = norm_by_axis(V)
        log.info(f'OPTIMIZER - Iter {iter_+1}: Error {F[-1]}')#', {overall_check(E,X, U, V, lambda_)}')#, Max Norm: {max(norm_U)}, Mean Norm: {np.mean(norm_U)}, Min Norm of U: {min(norm_U)}')
        #log.info(f'Iter {iter_+1}: Error {F[iter_]}, Max Norm: {max(norm_V)}, Mean Norm: {np.mean(norm_V)}, Min Norm of U: {min(norm_V)}')
        if iter_ > 10 and (np.array(F[-10:-1]).mean()-F[-1])/F[-1] < 0.0001:
            log.info(f'OPTIMIZER - Converged after {iter_} iterations')
            break
        elif iter_ >= max_iter -1:
            log.info(f'OPTIMIZER - Exit at max iter {max_iter}')
            break
        else:
            iter_ += 1
    norm_array = norm_by_axis(V)/norm_by_axis(U)
    mu = ((lambda_ * U * norm_array + E@(X - U@V.T)@V) * (U > 0)).sum(axis = 1)/(U>0).sum(axis = 1)
    #mu = np.array([sum([(lambda_*U[i,j] * norm_array[j] + (2*(X-U@V.T)@V)[i,j]) 
    #                for j in range(s)]*(U[i,:]>0))/sum(U[i,:]>0) for i in range(d)])
    mu = np.reshape(mu, (-1,1))
    return U,V,mu,F

def palm(E, P, U, V, lambda_ = 1e-06, eps = 1e-14, gamma1 = 1.1, gamma2 = 1.2 ):
    m,n = U.shape
    c =1/(gamma1 * (linalg.norm(E**2)*linalg.norm(V.T@V) + lambda_/eps))
    d = 1/(gamma2 * (linalg.norm(U.T@E**2@U) + lambda_*np.sqrt(m)*linalg.norm(U)))
    #log.info(f'c: {c}, d:{d}')
    log.debug(f'min norm U {min(norm_by_axis(U))} and min norm V {min(norm_by_axis(V,axis = 1))}')
    #time0 = time.time()
    F_U = -E**2@(P - U@V.T)@V + lambda_ * U @ np.diag([linalg.norm(V[:,j])/linalg.norm(U[:,j]) for j in range(n)])
    #time1 = time.time()
    U = projection(U - c * F_U, axis = 0)
    assert all(np.isclose(U@np.ones(n), np.ones(m)))
    #time2 = time.time()
    F_V = -(P - U@V.T).T@E**2@U + lambda_ * V @ np.diag([linalg.norm(U[:,j])/linalg.norm(V[:,j]) for j in range(n)])
    #time3 = time.time()
    V = projection(V - d * F_V, axis = 1)
    assert all(np.isclose(V.T@np.ones(m), np.ones(n)))
    #time4 = time.time()
    #log.debug(f'Time: {(time1 - time0, time2-time1, time3-time2, time4-time3)}')
    return U,V

def reduce(U,V, drop_threshold = 1e-8):
    norm = linalg.norm(U, ord = 2, axis = 0)
    drop = np.where(norm<drop_threshold)[0]
    if len(drop) > 0 :
        log.info(f'REDUCER - Drop columns {drop} where column norm is {norm[drop]}')
    #else:
    #    log.info(f'No reduction. Rank is still {U.shape[1]}')
    U = np.delete(U, drop, axis = 1)
    V = np.delete(V, drop, axis = 1)
    return U,V,U.shape[1] - len(drop)

def compress(U, V):
    d,s = U.shape
    combination = [[U[j,i] * V[k,i] for i in range(s)] for j in range(d) for k in range(d)]
    L,D,R = svd(combination)
    log.info(f'COMPRESSER - Can drop {sum(np.isclose(0,D))} columns')
    return U, V
    
def compress2(U, V, ind_threshold):
    pass
      
#TODO: Check all np.ones
def expand2(U,V,u,v,kappa):
    d,s = U.shape
    #kappa = 0.5**15/np.asscalar(max(abs(u)))
    U = np.diag(np.reshape(np.ones((d,1)) - kappa * u,-1))@U
    U = np.concatenate((U, kappa * u), axis = 1)
    V = np.concatenate((V, 1/(v.T@np.ones((d,1)))*v), axis = 1)
    #log.info(f'Expanded: Number of columns = {U.shape[1]}')
    return U,V

def expand(U,V,u,v,kappa):
    #pdb.set_trace()
    d,s = U.shape
    #kappa = 0.5**15/np.asscalar(max(abs(u)))
    U = np.diag(np.ones(d) - (kappa@u).flatten())@U
    try: 
        U = np.concatenate((U, kappa @ u), axis = 1)
    except:
        pdb.set_trace()
    #print(U.shape)
    assert validate(U)
    V = np.concatenate((V, 1/(v.T@np.ones((d,1)))*v), axis = 1)
    #log.info(f'Expanded: Number of columns = {U.shape[1]}')
    return U,V

def positive_normalize(x):
    return normalize((positify(x)), axis = 0)

def positify(x):
    return x * (x>0)

def check_global_optimality(E,P,X,mu, threshold = 10e-6, max_iter = 100, lr = 1000, extra_eps = 10e-7, lambda_ = 1e-6):
    logging.debug(f'BACKTRACKER - (threshold , max_iter, lr, extra_eps, lambda_): {(threshold , max_iter, lr, extra_eps, lambda_)}')
    d,s = X.shape
    t = mu@np.ones((1,d)) - E@(X - P)
    u = normalize(np.random.rand(d,1), axis = 0)
    v = normalize(np.random.rand(d,1), axis = 0)
    #assert and(np.isclose(norm_by_axis(u),1)[0],np.isclose(norm_by_axis(v),1)[0])
    iter_ = 0
    new_sigma = np.asscalar(u.T@t@v)
    log.info(f'BACKTRACKER - Initial sigma: {new_sigma}')
    while True:
        old_sigma = new_sigma
        u = positive_normalize(u + lr * t@v)
        v = positive_normalize(v + lr * t.T@u)
        new_sigma = np.asscalar(u.T@t@v)
        change = abs(new_sigma-old_sigma)/abs(old_sigma)
        #log.info(f'BACKTRACKER - Check global optimality: Iter = {iter_}, value = {new_sigma}, change = {change}')
        if change<threshold and iter_ > 3:
            log.info(f'BACKTRACKER - Converged: Iter = {iter_}, Sigma = {new_sigma}, Smaller than lambda: {new_sigma < lambda_}')
            log.info(f'BACKTRACKER - Quantile of u {np.quantile(u, q = [0.9,0.75,0.5,0.25,0.1])}')
            break
        elif iter_>=max_iter:
            log.info(f'BACKTRACKER - Escaped: Iter_ {iter_}, Sigma = {new_sigma}, Smaller than lambda: {new_sigma < lambda_}')
            break
        else:
            iter_ += 1
    return new_sigma < (1+extra_eps) * lambda_, u, v

def check_global_optimality2(E,P,X,mu, threshold = 10e-6, max_iter = 100, lr = 1000, extra_eps = 10e-7, lambda_ = 1e-6):
    logging.debug(f'BACKTRACKER - (threshold , max_iter, lr, extra_eps, lambda_): {(threshold , max_iter, lr, extra_eps, lambda_)}')
    d,s = X.shape
    t = mu@np.ones((1,d)) - E@(X - P)
    pdb.set_trace()
    v_sample = [normalize(np.random.rand(d,1), axis = 0) for _ in range(10000)]
    check = np.array([linalg.norm(positify(t@v)) for v in v_sample])
    if any(check > lambda_):
        temp = np.argmax(check)
        v = v_sample[temp]
        u = positive_normalize(t@v)
        log.info(f'BACKTRACKER - Sigma = {np.asscalar(u.T@t@v)}, Smaller than lambda: {np.asscalar(u.T@t@v) < lambda_}')
        log.info(f'BACKTRACKER - Quantile of u {np.quantile(u, q = [0.9,0.75,0.5,0.25,0.1])}')
        return False, u,v
    else:
        return True, None, None

def norm_by_axis(X,axis = 0):
    return linalg.norm(X, ord = 2, axis = axis)

def projection_single(X):
    d = len(X)
    temp = sorted(X)
    i = d-1
    while i>=0:
        if i == 0:
            t = (sum(X)-1)/d
            break
        else:
            t = (sum(temp[i:])-1)/(d-i)
            if t > temp[i-1]:
                break
            else:
                i -= 1
    X = (X - t * np.ones(d))
    return X*(X>0)

def projection(X, axis = 0):
    d,s = X.shape
    if axis == 0:
        proj = np.array([projection_single(X[i,:]) for i in range(d)])
    elif axis ==1:
        proj = np.array([projection_single(X[:,i]) for i in range(s)]).T
    return proj

def projection2(X, axis = 0):
    #pdb.set_trace()
    d,s = X.shape
    #print(X@np.ones(X.shape[1]))    
    proj = np.zeros([d,s])
    if axis == 0:
        for i in range(d):
            temp = sorted(X[i,:], reverse = True)
            l = np.argmax([sum([temp[k] - temp[j] for k in range(j)]) for j in range(s)])
            eta = 1/l * (1 - sum(temp[:l+1]))
            proj[i,:] = X[i,:] + eta * np.ones(s)
            proj[i,:] = proj[i,:] * (proj[i,:]>0)
        return proj
    elif axis == 1:
        for i in range(s):
            temp = sorted(X[:,i], reverse = True)
            l = np.argmax([sum([temp[k] - temp[j] for k in range(j)]) for j in range(d)])
            #pdb.set_trace()
            eta = 1/l * (1 - sum(temp[:l+1]))
            proj[:,i] = X[:,i] + eta * np.ones(d)
            proj[:,i] = proj[:,i] * (proj[:,i]>0)
        return proj
        

def init(d, s0, seed = 0):
    np.random.seed(seed)
    U = np.random.rand(d,s0)
    V = np.random.rand(d,s0)
    U = normalize(U, axis = 1, norm = 'l1')
    V = normalize(V, axis = 0, norm = 'l1')
    return U,V

if __name__ == "__main__":
    U0,V0 = init(100,5)
    P = U0 @ V0.T
    U1, V1 = init(100,10)
    #U,V,mu, F = subroutine(P,U1,V1,max_iter = 100, lambda_ = 0.01)
    U,V,mu, F = subroutine(P,U0,V0,max_iter = 100, lambda_ = 0.0001, tol = 0.0001)
    check_global_optimality(P, U0@V0.T, mu)
