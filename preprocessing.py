from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import cascaded_union
import numpy as np
import time
from util import *
import multiprocessing as mp
import sys 


if __name__ == "__main__":
    time0 = time.time()
    funclist = []
    transitions = []
    tax = pd.read_csv(r'yellow_tripdata_2016-01.csv',
                    usecols=['dropoff_longitude', 'dropoff_latitude', 'pickup_longitude', 'pickup_latitude'],
                    chunksize = 100000, dtype =np.float32)
    i = 0
    man = get_zones()
    joint = get_joint_grid(man, 100)
    pool = mp.Pool(4)
    for tax_sample in tax:
        f = pool.apply_async(process, args = [tax_sample], kwds = {'zones':joint})
        funclist.append(f)
    for f in funclist:
        transitions.append(f.get(timeout = 60))
        #Cannot dO this, why?
        #result.to_csv(r'test2.csv', index = False)
        i += 1
        print(f'Complete task {i}')
    pd.concat(transitions).to_csv(r'100.csv', index = False)
    print(f'Time:{time.time() - time0}')