import numpy as np
from sklearn.decomposition import nmf
from sklearn.preprocessing import normalize
from numpy import linalg, random
from numpy.linalg import norm
from util import *
from nmf_adaptive import *
from sklearn.cluster import KMeans
import argparse
import hickle
from matplotlib import pyplot as plt
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('statemap')
    parser.add_argument('matrix')
    parser.add_argument('grid')
    parser.add_argument('lambda_', default = 1e-10, type = float)
    parser.add_argument('eps', default = 1e-14, type = float)
    parser.add_argument('r', default = 5, type = int)
    parser.add_argument('kappa_threshold', default = 1e-4, type = float)
    parser.add_argument('starting_kappa', default = 0.5, type = float)
    parser.add_argument('starting_p', default = 1, type = int)
    parser.add_argument('outputpath',default = './', type = str)
    args = parser.parse_args()

    logger = logging.getLogger('runner')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{args.outputpath}runner.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #formatter = logging.Formatter('[%(levelname)7s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f'First load: Matrix={args.matrix}, Statemap={args.statemap}, Grid={args.grid}')
    P = hickle.load(args.matrix)
    statemap = hickle.load(args.statemap)
    grid = pd.read_csv(args.grid)
    
    logger.info(f'Run optimizer: lambda_={args.lambda_}, eps={args.eps},r={args.r}')
    d = P.shape[0]
    U,V,mu,E = non_negative_factorization(P = P,s0 = 1,lr = 100, starting_kappa = args.starting_kappa, starting_p = args.starting_p, kappa_threshold = args.kappa_threshold, max_iter=args.r, lambda_=args.lambda_, eps = args.eps, gamma1 = 1.1, gamma2 = 1.2, capture = True, outputpath = args.outputpath)
