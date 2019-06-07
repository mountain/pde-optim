#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import flow_tools
from numpy.random import random

root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)


# Aspect ratio

Lx, Ly = (6., 6.)
nx, ny = (320, 320)

# Create bases and domain

x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.Chebyshev('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Equations

Reynolds = 1e3
Schmidt = 1.

problem = de.IVP(domain, variables=['p', 'u', 'v', 'uy', 'vy', 's', 'sy'])
problem.meta[:]['y']['dirichlet'] = True
problem.parameters['Re'] = Reynolds
problem.parameters['Sc'] = Schmidt
problem.add_equation("dt(u) + dx(p) - 1 / Re * (dx(dx(u)) + dy(uy)) = - u * dx(u) - v * uy")
problem.add_equation("dt(v) + dy(p) - 1 / Re * (dx(dx(v)) + dy(vy)) = - u * dx(v) - v * vy")
problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(s) - 1/(Re * Sc) * (dx(dx(s)) + dy(sy)) = - u * dx(s) - v * sy")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("sy - dy(s) = 0")

problem.add_bc("left(u) = 0.5")
problem.add_bc("right(u) = -0.5")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("left(p) = 0", condition="(nx == 0)")
problem.add_bc("left(s) = 0")
problem.add_bc("right(s) = 1")

# Timestepping and solver

ts = de.timesteppers.RK443
solver = problem.build_solver(ts)

# Setup IVP

x = domain.grid(0)
y = domain.grid(1)
u = solver.state['u']
uy = solver.state['uy']
v = solver.state['v']
vy = solver.state['vy']
s = solver.state['s']
sy = solver.state['sy']

a = 0.1
sigma = 0.2
flow = -0.3
amp = -0.2

u['g'] = flow * np.tanh(y / a)
v['g'] = amp * np.sin(2.0 * np.pi * x / Lx * 6) * np.exp(- (y * y) / (sigma * sigma)) * np.exp(- (x * x / 36)) * (1 + 0.1 * random(y.shape))
s['g'] = 0.5 * (1 + np.tanh(y / a))

u.differentiate('y', out=uy)
v.differentiate('y', out=vy)
s.differentiate('y', out=sy)

solver.stop_sim_time = 20.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.2 * Lx / nx
cfl = flow_tools.CFL(solver, initial_dt, safety=0.8)
cfl.add_velocities(('u', 'v'))

# Make plot of scalar field
x = domain.grid(0, scales=domain.dealias)
y = domain.grid(1, scales=domain.dealias)
xm, ym = np.meshgrid(x, y)
fig, axis = plt.subplots(figsize=(Lx, Ly))
p = axis.pcolormesh(xm, ym, s['g'].T, cmap='RdBu_r')
axis.set_xlim([-Lx/2, Lx/2])
axis.set_ylim([-Ly/2, Ly/2])

logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 10 == 0:
        p.set_array(np.ravel(s['g'][:-1,:-1].T))
        plt.savefig('images/kh-%04d.png' % (solver.iteration // 10))
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)