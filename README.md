# HessianFreeOptimization
Course project for 10-725 convex optimization in Fall 2016.

See ... for our final report.

The code compare the convergence rate from the aspective of epoch, number of evaluation of objective function and gradient of modern first-order optimization method, GD with backtracking, L-BFGS and Hessian-free method.
We proposed a modification, which applies momentum to the L-BFGS method, making the L-BFGS compatible with the modern deep neural network paradigm, and at the same time owns the advantage of accelerated volocity in preceeding direction.
