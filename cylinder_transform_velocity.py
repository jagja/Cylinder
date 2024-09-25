"""
Version 19/07/2024

Serialises grid (integer) and velocity 
field (single precision float) from .xy
input file."""

import os
import argparse
import time
import numpy as np

def f_part(a):
    """Returns fractional part of input vector"""
    return a-np.floor(a)

def i_part(a):
    """Returns integer part of input vector"""
    return a.astype(int)

def to_uint(a):
    """Returns grid mapped to unsigned integer"""
    a = np.subtract(a, np.full(len(a),np.amin(a)))
    return np.around(a).astype(int)

def get_grid(a):
    """Returns scalar decomposition of coordinate set"""
    return np.unique(a[:,0]), np.unique(a[:,1]), np.unique(a[:,2])

def get_diff(a):
    """Returns grid spacing"""
    return np.round(np.mean(a[1:]-a[:-1]), 8)

def get_coord(a, s):
    """Returns nondimensional coordinate array"""
    return np.append(to_uint(np.divide(np.asarray(a), s)), -1)

def get_field(a, s):
    """Returns nondimensional velocity field array"""
    if np.abs(s) > 1e-16:
        return np.append(np.divide(np.asarray(a), s), 0.0)
    return np.append(np.asarray(a), 0.0)

def get_files(a):
    """Returns list of .xy files from subdirectories"""
    l = []
    os.chdir(a)
    for r, _, f in os.walk(os.getcwd()):
        for n in f:
            l.append(os.path.join(r, n))
    os.chdir('..')
    return sorted(l)

def get_parser():
    """Generate parser"""
    p = argparse.ArgumentParser(description='calculate LAVD field')
    p.add_argument('-d','--time_step', help="time difference",
                   type=float, required=True)

    p.add_argument('-i', '--time_initial', help="lower time integration limit",
                   type=float, required=True)

    p.add_argument('-f', '--time_final', help="upper time integration limit",
                   type=float, required=True)

    p.add_argument('-n', '--dir_name', help="directory containing fields",
                   required=True)

    return p

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    dt = args.time_step
    dname = args.dir_name

    t_ex = -time.time()
    t_i = int(args.time_initial/dt)
    t_f = int(args.time_final/dt)

    files = get_files(dname)
    if not os.path.isdir('Velocity'):
        os.mkdir('Velocity')

    uvw_arr = np.loadtxt(files[0])

    x_grid, y_grid, z_grid = get_grid(uvw_arr)
    n_x, n_y, n_z = len(x_grid), len(y_grid), len(z_grid)
    dx, dy, dz = get_diff(x_grid), get_diff(y_grid), get_diff(z_grid)

    position = np.array([get_coord(uvw_arr[:,0], dx),
                         get_coord(uvw_arr[:,1], dy),
                         get_coord(uvw_arr[:,2], dz)],
                        dtype=np.ushort)
    np.save("Velocity\\pos", position)

    for file in files[t_i:t_f]:
        uvw_arr = np.loadtxt(file)

        velocity = np.array([get_field(uvw_arr[:,3], dx),
                             get_field(uvw_arr[:,4], dy),
                             get_field(uvw_arr[:,5], dz)],
                            dtype=np.float32)

        np.save("Velocity\\vel_" + str(files.index(file)), velocity)
        print("\033[K"+"Serialised array from", file, end="\r")

    t_ex += time.time()
    print(f"\nExecuted in {t_ex:.2f} seconds.")
