"""
Jhonatan da Silva
Last Updated version :
Thu Feb  2 09:30:58 2017
Number of code lines: 
30
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import random

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

class gradientDescent():

    def __init__(self):
        self.x = np.linspace(-10,10,100)
        self.y = self.x**3
        self.xdot = random.choice(self.x) 
        self.ydot = 0
        p

    def animate(self,i):
        self.ydot = self.xdot**3
        step = 0.1
        ax1.clear()
        plt.plot(self.x,self.y)
        plt.plot(self.xdot,self.ydot,marker='o')
        self.xdot += step

gradient = gradientDescent() 
ani = animation.FuncAnimation(fig,gradient.animate)
plt.show()
