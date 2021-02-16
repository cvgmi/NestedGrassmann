import numpy as np
import pymanopt
from pymanopt.solvers.solver import Solver
from pymanopt.solvers.steepest_descent import SteepestDescent

def compute_centroid(manifold, points):
    """Compute the centroid of `points` on the `manifold` as Karcher mean."""
    num_points = len(points)

    @pymanopt.function.Callable
    def objective(y):
        accumulator = 0
        for i in range(num_points):
            accumulator += manifold.dist(y, points[i]) ** 2
        return accumulator / 2

    @pymanopt.function.Callable
    def gradient(y):
        g = manifold.zerovec(y)
        g = g.astype(points.dtype)
        for i in range(num_points):
            g -= manifold.log(y, points[i])
        return g

    # XXX: Manopt runs a few TR iterations here. For us to do this, we either
    #      need to work out the Hessian of the Karcher mean by hand or
    #      implement approximations for the Hessian to use in the TR solver as
    #      Manopt. This is because we cannot implement the Karcher mean with
    #      Theano, say, and compute the Hessian automatically due to dependence
    #      on the manifold-dependent distance function, which is written in
    #      numpy.
    solver = SteepestDescent(maxiter=15)
    problem = pymanopt.Problem(manifold, objective, grad=gradient, verbosity=0)
    return solver.solve(problem)
