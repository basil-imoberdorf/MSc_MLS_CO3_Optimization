from sklearn.datasets import load_iris
from FunLibLogReg import *

# Get the iris dataset
iris = load_iris()

# Extract the petal length of the Versicolor and Virginica flowers
Petal_Length = iris.data[50:150, 2] - np.mean(iris.data[50:150, 2])
# Assign zeros to the Versicolor flowers and ones to the Virginica flowers
Target_Label = np.hstack((np.zeros(50), np.ones(50)))

# Plot the loss landscape for visualization
a_lim = [-300, 300]
m_lim = [-300, 300]
plotLossLandscape(Petal_Length, Target_Label, a_lim, m_lim, 300)

########################################################################################################################
# Apply Gradient Descent
iters = 100
initial_guess = [-25, -1]  # [a0, m0]
learning_rate = shapeSigmoid(iters, 2, 0.1, 2)  # learning rate is a function of number of iterations
[popt, min_SSE, a_vec, m_vec, SSE_vec] = GradientDescent(x=Petal_Length,
                                                         y=Target_Label,
                                                         param_init=initial_guess,
                                                         iters=iters,
                                                         learning_rate=learning_rate,
                                                         normalize=1)

print('Best value found for the slope:     a =', round(popt[0], 4))
print('Best value found for the intercept: m =', round(popt[1], 4))
print('With a minimal loss of:  SSE =', round(min_SSE, 3))
# Calculate how good classifier fitted with gradient descent is
getPerformance(Petal_Length, Target_Label, popt[0], popt[1], 0.5)
print("%------------------------------------------------------")

# plot evolution of loss
plt.close()
plt.plot(np.arange(iters+1), SSE_vec, linewidth=2)
plt.xlim(0, iters)
plt.grid(alpha=0.3)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Sum of Squares Error', fontsize=14)
plt.title('Loss evolution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# plot evolution of loss on the space spanned by the parameters
a_lim = [-30, 30]
m_lim = [-20, 30]
resolution = 200
[A, M, SSE] = getRaster(Petal_Length, Target_Label, a_lim, m_lim, resolution)

plt.close()
contour = plt.contour(A, M, SSE, cmap='jet', levels=200, alpha=0.7)
scatter = plt.scatter(a_vec, m_vec, c=np.arange(len(a_vec)), cmap='inferno', s=100, marker='+', zorder=10)
plt.colorbar(scatter, pad=0.01, label='Iterations')
plt.xlim(a_lim[0], a_lim[1])
plt.ylim(m_lim[0], m_lim[1])
plt.grid(alpha=0.3)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlabel('Slope - a', fontsize=14)
plt.ylabel('Intercept - b', fontsize=14)
plt.title('Visualization of Gradient Descent', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


########################################################################################################################
# Plot the classes and the curves of best fit
plt.scatter(Petal_Length[:49], Target_Label[:49], c='red', s=100, edgecolor='k', label='Iris Versicolor', alpha=0.5)
plt.scatter(Petal_Length[50:], Target_Label[50:], c='blue', s=100, edgecolor='k', label='Iris Virginica', alpha=0.5)

# Get optimal parameter values at a tolerance of 1e-10 using downhill simplex method initialized close to the optimum
print('\n%------------------------------------------------------')
print('Perform optimization with Sci-Py near the optimum to get a reference for what the optimum is\n')
initial_guess = [10, 0]  # [a0, m0]
result = SciPyMinimize(Petal_Length, Target_Label, initial_guess)
a_opt = result.x[0]
m_opt = result.x[1]
print('The optimal value for the slope is:     a =', round(a_opt, 4))
print('The optimal value for the intercept is: m =', round(m_opt, 4))

# Plot the optimal curve as benchmark reference for the fit of the gradient descent algorithm
x_Ref = np.linspace(-3, 3, 200)
plt.plot(x_Ref, 1/(1 + np.exp(-(a_opt*x_Ref + m_opt))), c='black', alpha=1, linewidth=1.5, label='SciPy model')
plt.plot(x_Ref, 1/(1 + np.exp(-(popt[0]*x_Ref + popt[1]))), c=(0.35, 0, 1), alpha=0.5, linewidth=4, label='GD model')

# Add labels and legend
plt.xlabel('Petal Length [cm] (mean centered)', fontsize=14)
plt.ylabel('Probability for Virginica', fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.xlim(-3, 3)
plt.ylim(-0.05, 1.05)
plt.grid(zorder=0)
plt.tight_layout()
plt.show()

# Calculate how good the with SciPy fitted classifier is
getPerformance(Petal_Length, Target_Label, a_opt, m_opt, 0.5)
