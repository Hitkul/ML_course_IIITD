#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, X,Y):
    totalError= sum((Y - (m * X + b)) ** 2)
    return totalError / float(X.shape[0])

def step_gradient(b_current, m_current, X,Y, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(X.shape[0])
    b_gradient += -(2/N) * sum(Y - ((m_current * X) + b_current))
    m_gradient += -(2/N) * sum(X * (Y - ((m_current * X) + b_current)))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(X,Y, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, X,Y, learning_rate)
    return [b, m]


points = genfromtxt("data.csv", delimiter=",")
X,Y = points[:,0], points[:,1]
learning_rate = 0.0001
initial_b = 0 # initial y-intercept guess
initial_m = 0 # initial slope guess
num_iterations = 1000
print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, X,Y)))
print("Running...")
[b, m] = gradient_descent_runner(X,Y, initial_b, initial_m, learning_rate, num_iterations)
print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, X,Y)))

