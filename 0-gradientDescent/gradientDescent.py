"""
Jhonatan da Silva
Last Updated version :
Thu Feb  2 11:08:49 2017
Number of code lines: 
41
"""
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import bokeh.plotting as bp
from matplotlib import style
import numpy as np
import random

#style.use('fivethirtyeight')
class gradientDescent():

    def __init__(self):
        self.x = np.linspace(-10,10,100)
        self.y = self.x**2
        self.xdot = random.choice(self.x) 
        self.ydot = 0
        self.mins = []
        self.cost = []

    def derivative(self,x):
        #test function = x^2
        return 2*x
    
    def GD(self):
        print('Initializing Gradient Descent')
        oldMin = 0
        currentMin = 7
        #precision
        epsilon = 0.001
        step = 0.01
        while abs(currentMin - oldMin) > epsilon:
            oldMin = currentMin
            gradient = self.derivative(oldMin)
            move = gradient * step
            currentMin = oldMin - move
            self.cost.append((3-currentMin)**2)
            self.mins.append(currentMin)
        print('Local min : {:.2f}'.format(currentMin))
    
gradient = gradientDescent() 
gradient.GD()
