"""
Version 27/07/2024

Calculates trajectories of passive Lagrangian tracers for given
velocity field using a fourth order Runge-Kutta
method and trilinear interpolation.

Uses trajectories to evaluate Lagrangian averages of
vorticity deviation.

"""

import heapq
import argparse
import re
import os
import time
import numpy as np
import cupy as cp

def f_part(a):
    """Returns fractional part of input vector"""
    return cp.subtract(a, cp.floor(a))

def i_part(a):
    """Returns integer part of input vector"""
    return a.astype(cp.int32)

def to_uint(a):
    """Returns grid mapped to unsigned integer"""
    a = cp.subtract(a, cp.full(len(a),cp.amin(a)))
    return cp.around(a).astype(cp.int32)

def get_grid(a):
    """Returns scalar decomposition of coordinate set"""
    return np.unique(a[0,:]), np.unique(a[1,:]), np.unique(a[2,:])

def get_diff(a):
    """Returns grid spacing"""
    return np.round(np.mean(a[1:]-a[:-1]), 8)

def get_coord(a, s):
    """Returns nondimensional coordinate array"""
    return cp.append(to_uint(cp.divide(cp.asarray(a), s)), -1)

def get_field(a, s):
    """Returns nondimensional velocity field array"""
    return cp.append(cp.divide(cp.asarray(a), s), 0.0)

def get_config(a, b, c, r):
    """Returns grid configuration for locating nearest neighbours"""
    g = cp.arange(r.shape[1]-1)
    m = cp.negative(cp.ones((a + 2, b + 2, c + 2), dtype=cp.int32))
    m[cp.add(r[0, :-1], 1),
      cp.add(r[1, :-1], 1),
      cp.add(r[2, :-1], 1)] = g[:]
    return m

def get_files(a):
    """Returns list of .xy files from subdirectories"""
    c = os.getcwd()
    l = []
    os.chdir(a)
    for r, _, f in os.walk(os.getcwd()):
        for n in f:
            l.append(os.path.join(r, n))
    os.chdir(c)
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

    p.add_argument('-w', '--time_window', help="time integration window",
                        type=float, required=True)

    p.add_argument('-x', '--x_subgrid', help="subgrid resolution (axial)",
                        type=int, required=False, default = 1)

    p.add_argument('-t', '--t_subgrid', help="subgrid resolution (temporal)",
                        type=int, required=False, default = 8)
    return p

def sort_files(a):
    """Sort files according to time index"""
    l = []
    for f in a:
        k = re.findall(r'\d+', f)
        if len(k):
            heapq.heappush(l, (int(k[-1]), f))
    return [e[1] for e in sorted(l)]

def initialise_field(f):
    """Returns initial velocity array"""
    q = np.load(f[0]).shape
    u = cp.zeros((len(f), q[0], q[1]), dtype=cp.float32)
    for k, s in enumerate(f):
        u[k, :] = cp.array(np.single(np.load(s)))
        print("Loaded", str(k), "snapshots from", s, end="\r")
    print()
    return u

def initialise_device():
    """Deallocate device memory"""
    m = cp.get_default_memory_pool()
    p = cp.get_default_pinned_memory_pool()
    m.free_all_blocks()
    p.free_all_blocks()

def initialise_coord(g, d, e):
    """Returns deterministic (grid)"""
    a = cp.square(cp.subtract(g[1], cp.median(g[1])))
    b = cp.square(cp.subtract(g[2], cp.median(g[2])))
    s = cp.multiply(1.0-e, cp.amax(cp.sqrt(a)))
    t = cp.equal(np.int32(d), g[0])
    k = cp.less(cp.sqrt(cp.add(a,b)), s)
    c = cp.where(cp.logical_and(t, k))[0]
    r, p, q = cp.full(len(g[0, c]),d), g[1, c], g[2, c]
    return cp.array([r, p, q])

def lerp(a, b, t):
    """Returns the linear interpolation between two vectors"""
    return cp.add(a, cp.multiply(t,cp.subtract(b,a)))

def hermite(a, b, c, d, t):
    """Two point hermite interpolation"""
    w_0 = 2.0*t**3 - 3.0*t**2 + 1.0
    w_1 = -2.0*t**3 + 3.0*t**2
    w_2 = t**3 - 2.0*t**2 + t
    w_3 = t**3 - t**2

    p = cp.multiply(a, w_0)
    p = cp.add(p, cp.multiply(b, w_1))
    p = cp.add(p, cp.multiply(c, w_2))
    p = cp.add(p, cp.multiply(d, w_3))
    return p

def lagrange(a, b, c, t):
    """Four point lagrange interpolation"""
    w_0 = 2.0*t**2 - 3.0*t + 1.0
    w_1 = 4.0*(-t**2 + t)
    w_2 = 2.0*t**2 - t

    p = cp.multiply(a, w_0)
    p = cp.add(p, cp.multiply(b, w_1))
    p = cp.add(p, cp.multiply(c, w_2))
    return p

def trilinear(a, b, c, s):
    """Returns three dimensional linear interpolation values"""
    s_00 = lerp(s[0,:], s[4,:], a)
    s_01 = lerp(s[1,:], s[5,:], a)
    s_10 = lerp(s[2,:], s[6,:], a)
    s_11 = lerp(s[3,:], s[7,:], a)

    s_0 = lerp(s_00, s_10, b)
    s_1 = lerp(s_01, s_11, b)

    return lerp(s_0, s_1, c)

def eval_mean(w, d):
    """Interpolate average vorticity (lagrange quadratic) values"""
    v = cp.zeros((d+1,3), dtype=cp.float32)
    v[0,:] = cp.mean(w[0,:], axis=1, keepdims=True).ravel()
    for j in range(1,d):
        q = cp.reshape(lagrange(w[0, :],
                                w[1, :],
                                w[2, :], j*1.0/d), (3, w.shape[2]))
        v[j,:] = cp.mean(q, axis=1, keepdims=True).ravel()
    v[d, :] =  cp.mean(w[-1,:], axis=1, keepdims=True).ravel()
    return v

#@profile
def eval_rhs(r, m, s):
    """Returns right hand side of differential equation"""
    i, j, k = i_part(r[0, :]), i_part(r[1, :]), i_part(r[2, :])
    i_p, j_p, k_p = cp.add(i, 1), cp.add(j, 1), cp.add(k, 1)

    f = cp.take(s, [m[i,j,k],
                    m[i,j,k_p],
                    m[i,j_p,k],
                    m[i,j_p,k_p],
                    m[i_p,j,k],
                    m[i_p,j,k_p],
                    m[i_p,j_p,k],
                    m[i_p,j_p,k_p]])

    return trilinear(f_part(r[0, :]), f_part(r[1, :]), f_part(r[2, :]), f)

def eval_ivd(r, m, s, c):
    """Evaluate the instantaneous vorticity deviation"""
    w = cp.array([eval_rhs(r, m, s[0,: ]),
                  eval_rhs(r, m, s[1,: ]),
                  eval_rhs(r, m, s[2,: ])])
    return cp.linalg.norm(cp.subtract(w, c), axis=0)

def plot_field(r, f):
    """Prepares a 2D slice of field variable for plotting"""
    p, q = np.int32(cp.asnumpy(r[1])), np.int32(cp.asnumpy(r[2]))
    l = cp.asnumpy(f)
    m = np.zeros((np.amax(p+1),np.amax(q+1)))
    m[p,q] = l
    return m

def rk4_step(r, m, s, h):
    """Returns single integration step using fourth order Runge-Kutta method"""
    r_p = r
    k_1 = cp.multiply(h, cp.array([eval_rhs(r_p, m, s[0, 0,: ]),
                                   eval_rhs(r_p, m, s[0, 1,: ]),
                                   eval_rhs(r_p, m, s[0, 2,: ])]))

    r_p = cp.add(r, cp.multiply(0.5, k_1))
    k_2 = cp.multiply(h, cp.array([eval_rhs(r_p, m, s[1, 0,: ]),
                                   eval_rhs(r_p, m, s[1, 1,: ]),
                                   eval_rhs(r_p, m, s[1, 2,: ])]))

    r_p = cp.add(r, cp.multiply(0.5, k_2))
    k_3 = cp.multiply(h, cp.array([eval_rhs(r_p, m, s[1, 0,: ]),
                                   eval_rhs(r_p, m, s[1, 1,: ]),
                                   eval_rhs(r_p, m, s[1, 2,: ])]))

    r_p = cp.add(r, k_3)
    k_4 = cp.multiply(h, cp.array([eval_rhs(r_p, m, s[2, 0,: ]),
                                   eval_rhs(r_p, m, s[2, 1,: ]),
                                   eval_rhs(r_p, m, s[2, 2,: ])]))

    return cp.add(cp.divide(cp.add(k_1, k_4), 6.0), cp.divide(cp.add(k_2, k_3), 3.0))

def average_field(w, d):
    """Returns all trajectories for time t"""
    l = len(w)//2 - 1*(len(w)%2==0)
    b = cp.zeros((d*l+1, 3), dtype=cp.float32)
    for i in range(l):
        b[i*d:(i+1)*d+1, :] = eval_mean(w[2*i:2*i+3], d)
    return b[:, :, cp.newaxis]

def integrate(r, f, h):
    """Returns all trajectories for time t"""
    m, u, _, _ = f
    l = len(u)//2 - 1*(len(u)%2==0)
    a = cp.zeros((3*r.shape[1], l+1), dtype=cp.float32)
    a[:, 0] = cp.ravel(r)
    for i in range(l):
        dr = rk4_step(r, m, u[2*i:2*i+3], h)
        r = cp.add(r, dr)
        a[:, i+1] = cp.ravel(r)
    return a

def interpolate(r, m, s, w, c, h, d):
    """Interpolate position (hermite) and vorticity (lagrange quadratic) values"""
    a = cp.reshape(r[:,0], (3, r.shape[0]//3))
    b = cp.reshape(r[:,1], (3, r.shape[0]//3))

    u = cp.multiply(h, cp.array([eval_rhs(a, m, s[0, 0,: ]),
                                 eval_rhs(a, m, s[0, 1,: ]),
                                 eval_rhs(a, m, s[0, 2,: ])]))

    v = cp.multiply(h, cp.array([eval_rhs(b, m, s[2, 0,: ]),
                                 eval_rhs(b, m, s[2, 1,: ]),
                                 eval_rhs(b, m, s[2, 2,: ])]))

    f = eval_ivd(a, m, w[0,:], c[0,:])

    for j in range(1,d):
        x = cp.reshape(hermite(cp.ravel(a), cp.ravel(b),
        	                   cp.ravel(u), cp.ravel(v),
        	                   j*1.0/d), (3, r.shape[0]//3))

        q = cp.reshape(lagrange(w[0, :], w[1, :], w[2, :], j*1.0/d),
        	                    (3, w.shape[2]))

        f = cp.add(f, cp.multiply(2*(1+j%2), eval_ivd(x, m, q, c[j,:])))

    return cp.add(f, eval_ivd(b, m, w[2,:], c[d,:]))

def get_lavd(r, f, h, d):
    """Evaluate the Lagrange averaged vorticity deviation over a time window"""
    g = cp.zeros(r.shape[0]//3, dtype=cp.float32)
    m, s, w, c = f
    for i in range(r.shape[1]-1):
        v = interpolate(r[:,i:i+2], m, s[2*i:2*i+3], w[2*i:2*i+3], c[i*d:(i+1)*d+1], h, d)
        g = cp.add(g, cp.multiply(h/(3.0*d), v))
    return g

if __name__ == '__main__':
    initialise_device()

    parser = get_parser()
    args = parser.parse_args()
    dt = args.time_step

    t_ex = -time.time()
    t_i = int(args.time_initial/dt)
    t_f = int(args.time_final/dt)
    t_w = int(args.time_window/dt)

    r_t = args.t_subgrid
    r_x = args.x_subgrid

    files_1 = get_files('Temp\\Velocity')
    files_2 = get_files('Temp\\Vorticity')

    xyz = cp.array(np.load(files_1[0]))
    x_grid, y_grid, z_grid = get_grid(xyz)
    n_x, n_y, n_z = len(x_grid), len(y_grid), len(z_grid)

    config = get_config(n_x, n_y, n_z, xyz)
    r_0 = initialise_coord(xyz[:, :-1], 0, 0.05)

    set_1 = sort_files(files_1[1:])[t_i:t_f+t_w+1]
    set_2 = sort_files(files_2[1:])[t_i:t_f+t_w+1]
    velocity = initialise_field(set_1)
    vorticity = initialise_field(set_2)

    v_0 = average_field(vorticity, r_t)
    axis = np.linspace(5, n_x-5, 1 + r_x*(n_x-10))
    out_shape = (len(axis), r_0.shape[1], r_0.shape[0]+1)
    lavd_3d = cp.zeros(out_shape, dtype=cp.float32)

    for i_t in range(t_f-t_i+1):
        fields = (config,
                  velocity[i_t:i_t+t_w+1],
                  vorticity[i_t:i_t+t_w+1],
                  v_0)

        for j_x, x_const in enumerate(axis):
            print(f"Integrating from t = {dt*(i_t+t_i):.2f}",
                  f"until t = {dt*(i_t+t_i+t_w):.2f},",
                  f"x = {x_const:.2f}.", end="\r")
            r_0[0, :] = cp.full(r_0.shape[1], x_const)
            traj_2d = integrate(r_0, fields, 2.0*dt)
            lavd_2d = get_lavd(traj_2d, fields, 2.0*dt, r_t)
            lavd_3d[j_x, :] = cp.vstack((r_0, lavd_2d)).T

        lavd = cp.asnumpy(lavd_3d)
        np.save("lavd_"+str(i_t+t_i)+"_"+str(i_t+t_i+t_w), lavd)
    t_ex += time.time()
    print(f"\nExecuted in {t_ex:.2f} seconds.")
    #field = plot_field(r_0,lavd[60,:,3])[100:300, 100:300]
    #from matplotlib import pyplot as plt
