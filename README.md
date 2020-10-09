# Project 1: Reduced model approach
## Exercises

This file contains multiple solvers for an optimal control problem icluding
    - fast poisson solver framework to fastly calculate the inverse of our
    matrix system
    - stationary solver
    - damped jacobi solver
    - matrix free krylov method using the conjugated gradient method
    - multigrid with the stationary method and damped jacobi as smoothers

We wrote this file as a project of the lecture "iterative methods and 
preconditioning" from Jun.-Prof. Gabriele Ciaramelle, held in the WS2020/2021
at the University of Konstanz

The other according files contain multiple tests and functions to plot
the behaviour of our solvers for different parameters. We used them
to analyse their convergence behaviour. 

## Report

This folder includes a tex and pdf version of our report. In this small report we analyse the behaviour and the convergence rates of
all the algorithms named above and we provide some mathematical background information.
