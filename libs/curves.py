# coding=utf-8
import pyglet
from OpenGL.GL import (glUseProgram, glClearColor, glEnable, GL_DEPTH_TEST,
                       glUniformMatrix4fv, glGetUniformLocation, GL_TRUE,
                       glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       glPolygonMode, GL_FRONT_AND_BACK, GL_FILL, GL_LINES)
import numpy as np
import libs.transformations as tr

# Funciones de las curvas
def generateT(t):
    return np.array([[1, t, t**2, t**3]]).T

def hermiteMatrix(P1, P2, T1, T2):

    # Generate a matrix concatenating the columns
    G = np.concatenate((P1, P2, T1, T2), axis=1)

    # Hermite base matrix is a constant
    Mh = np.array([[1, 0, -3, 2],
                   [0, 0, 3, -2], 
                   [0, 1, -2, 1], 
                   [0, 0, -1, 1]])

    return np.matmul(G, Mh)


def bezierMatrix(P0, P1, P2, P3):

    # Generate a matrix concatenating the columns
    G = np.concatenate((P0, P1, P2, P3), axis=1)

    # Bezier base matrix is a constant
    Mb = np.array([[1, -3, 3, -1], 
                   [0, 3, -6, 3],
                   [0, 0, 3, -3], 
                   [0, 0, 0, 1]])

    return np.matmul(G, Mb)


# M is the cubic curve matrix, N is the number of samples between 0 and 1
def evalCurve(M, N):
    # The parameter t should move between 0 and 1
    ts = np.linspace(0.0, 1.0, N)

    # The computed value in R3 for each sample will be stored here
    curve = np.ndarray(shape=(N, 3), dtype=float)

    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T

    return curve

P1 = np.array([[0, 0, -5]]).T
P2 = np.array([[0, 0, 5]]).T
T1 = np.array([[60, 0, 0]]).T
T2 = np.array([[60, 0, 0]]).T

# Creamos la curva
#GMh = hermiteMatrix(P1, P2, T1, T2)

#HermiteCurve = evalCurve(GMh, N)

# Se pueden concatenar las 2 curvas de Bezier en una sola
#BezierCurve = np.concatenate((bezierCurve1, bezierCurve2), axis=0)



