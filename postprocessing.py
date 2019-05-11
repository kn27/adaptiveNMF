from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import cascaded_union
import numpy as np
import time
from sklearn.preprocessing import normalize
from numpy import linalg
from util import *
import hickle as hkl 
import pdb

def group_and_map(filename):
    t = pd.read_csv(filename)
    t['count'] = 1
    count = t.groupby(['pu','do'])['count'].count()
    count = pd.DataFrame(count)
    count.columns = ['Count']
    count.reset_index(inplace = True)
    del t
    keep = count.groupby(['pu'])['Count'].sum()
    keep = list(keep[keep/keep.sum() > 1e-4].index)
    count = count[count.pu.apply(lambda x: x in keep) * count.do.apply(lambda x: x in keep)]
    count.reset_index(inplace = True, drop = True)
    unique = count.pu.unique()
    statemap = dict(zip(unique, range(len(unique))))
    return count, statemap

def build_matrix(count, statemap):
    A = np.zeros((len(statemap),len(statemap)))
    for i in range(count.shape[0]):
        try:
            A[int(statemap[count.pu[i]]), int(statemap[count.do[i]])] = count.Count[i]
        except Exception as e:
            raise ValueError(e)
    A = normalize(A, norm = 'l1',axis = 1)
    assert all(np.isclose(linalg.norm(A, ord = 1,axis = 1), np.ones(A.shape[0])))
    return A

def inspect(A):
    plt.hist(A.sum(axis = 0), bins = 15,range = (0,15))
    plt.show()
    plt.hist(A.sum(axis = 1), bins = 15,range = (0,15))
    plt.show()

def get_center(grid):
    grid.geometry = grid.geometry.apply(lambda x: x.centroid)
    return grid

if __name__ == "__main__":
    man = get_zones()
    grid = get_joint_grid(man, 100, True)
    grid = get_center(grid)

    count,statemap = group_and_map('100.csv')

    zone = gpd.read_file(r'taxi_zones\taxi_zones.shp')
    zone = zone.to_crs("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    man= zone[zone.borough == 'Manhattan']
    fig, ax = plt.subplots(figsize = (10,10))
    man.plot(ax=ax)
    grid.loc[list(set(count[count.Count > 200].pu)),].plot(**{'edgecolor': 'yellow', 'alpha':0.9}, ax = ax)

    A = build_matrix(count,statemap)

    hkl.dump(A, '100_matrix.hkl' )