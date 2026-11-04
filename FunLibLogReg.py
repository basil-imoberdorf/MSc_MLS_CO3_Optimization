import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


########################################################################################################################
def getSSE(x, y, a, m):
    y_pred = 1 / (1 + np.exp(-(a*x + m)))
    SSE = 0.5*np.sum((y-y_pred)**2)
    return SSE, y_pred


########################################################################################################################
def GradientDescent(x, y, param_init, iters, learning_rate, normalize):
    start_time = time.time()  # measure the time it takes to complete the algorithm

    # Pre-allocate arrays to store values and for plotting
    a_Container = np.zeros(iters+1)  # its +1 because the initial values are like iteration step 0
    m_Container = np.zeros(iters+1)
    SSE_Container = np.zeros(iters+1)
    # Paste initial values
    a_Container[0] = param_init[0]
    m_Container[0] = param_init[1]

    # Initialize variables that keep track of the optimum values because they are not necessarily in the last iteration
    min_a = param_init[0]
    min_m = param_init[1]
    min_SSE = float('inf')  # make the first loss infinite

    # Apply GD for a fixed number of iterations
    for i in range(iters):
        # Evaluate the sigmoid function for the current parameter values
        [current_SSE, y_pred] = getSSE(x, y, a_Container[i], m_Container[i])
        SSE_Container[i] = current_SSE  # append the SSE for the current parameter values

        # Update if necessary the so far best found parameters
        if SSE_Container[i] < min_SSE:
            min_SSE = SSE_Container[i]
            min_a = a_Container[i]
            min_m = m_Container[i]

        # Calculate the gradient
        dSSEda = sum((y_pred - y)*y_pred*(1-y_pred)*x)
        dSSEdm = sum((y_pred - y)*y_pred*(1-y_pred))
        # Apply normalization if the argument "normalize" is enabled, i.e. set to 1 or True
        if normalize:
            Magnitude = np.sqrt(dSSEda**2 + dSSEdm**2)
            dSSEda = dSSEda/Magnitude
            dSSEdm = dSSEdm/Magnitude

        # move in the direction of the gradient
        a_Container[i+1] = a_Container[i] - learning_rate[i]*dSSEda
        m_Container[i+1] = m_Container[i] - learning_rate[i]*dSSEdm

    # Get the SSE of the last parameter values to complete the container
    [current_SSE, _] = getSSE(x, y, a_Container[-1], m_Container[-1])
    SSE_Container[-1] = current_SSE
    # Group the optimal parameters in an array
    param_opt = [min_a, min_m]

    # Print the elapsed time
    end_time = time.time()
    print("%------------------------------------------------------")
    print(iters, "iterations of Gradient Descent took:", round(end_time - start_time, 5), "seconds to compute\n")
    return param_opt, min_SSE, a_Container, m_Container, SSE_Container


########################################################################################################################
def SimulatedAnnealing(x, y, param_init, iters, step_size, Temp):
    start_time = time.time()  # measure the time it takes to complete the algorithm

    # Pre-allocate arrays to store values and for plotting
    a_Container = np.zeros(iters+1)  # its +1 because the initial values are like iteration step 0
    m_Container = np.zeros(iters+1)
    SSE_Container = np.zeros(iters+1)
    # Paste initial values
    a_Container[0] = param_init[0]
    m_Container[0] = param_init[1]
    [initial_SSE, _] = getSSE(x, y, a_Container[0], m_Container[0])
    SSE_Container[0] = initial_SSE

    # Initialize variables that keep track of the optimum values because they are not necessarily in the last iteration
    min_a = param_init[0]
    min_m = param_init[1]
    min_SSE = float('inf')  # make the first loss infinite

    np.random.seed(42)  # for reproducibility
    for i in range(iters):

        # Add a gaussian noise to the current parameter value to obtain a new ones
        a_trial = a_Container[i] + step_size[i]*np.random.randn()
        m_trial = m_Container[i] + step_size[i]*np.random.randn()

        # Evaluate the sigmoid function for the current parameter values
        [SSE_Container[i+1], _] = getSSE(x, y, a_trial, m_trial)

        if SSE_Container[i+1] <= SSE_Container[i]:  # if the new parameters reduce the SSE set them to be the next points
            a_Container[i+1] = a_trial
            m_Container[i+1] = m_trial

        elif np.exp(-(SSE_Container[i+1] - SSE_Container[i])/Temp[i]) >= np.random.rand():  # small chance to still accept
            a_Container[i+1] = a_trial
            m_Container[i+1] = m_trial

        else:  # otherwise paste the old parameters values to try again from there in the next iteration
            a_Container[i+1] = a_Container[i]
            m_Container[i+1] = m_Container[i]
            SSE_Container[i+1] = SSE_Container[i]

        # Update if necessary the so far best found parameters
        if SSE_Container[i] < min_SSE:
            min_SSE = SSE_Container[i]
            min_a = a_Container[i]
            min_m = m_Container[i]

    # Group the optimal parameters in an array
    param_opt = [min_a, min_m]

    # Print the elapsed time
    end_time = time.time()
    print("%------------------------------------------------------")
    print(iters, "iterations of Simulated Annealing took:", round(end_time - start_time, 5), "seconds to compute\n")
    return param_opt, min_SSE, a_Container, m_Container, SSE_Container


########################################################################################################################
def SciPyMinimize(x, y, initial_guess):
    def ObjFun(params, x, y):
        y_pred = 1/(1 + np.exp(-(params[0]*x + params[1])))
        return np.sum(0.5*(y - y_pred)**2)

    result = minimize(ObjFun,
                      initial_guess,
                      args=(x, y),
                      method="Nelder-Mead",
                      tol=1e-10,
                      options={'disp': True})
    return result


########################################################################################################################
# Note: offset should be in the interval [1,iters] and if slope is negative then the output will grow instead of fall
def shapeSigmoid(iters, ceil, slope, shift):
    x = np.arange(iters)
    return ceil/(1 + np.exp(slope*(x - iters/shift)))


########################################################################################################################
def getRaster(x, y, a_lim, m_lim, resolution):
    [A, M] = np.meshgrid(np.linspace(a_lim[0], a_lim[1], resolution), np.linspace(m_lim[0], m_lim[1], resolution))
    SSE = np.zeros((resolution, resolution))  # Pre-allocate array
    for i in range(resolution):
        for j in range(resolution):
            [SSE[i, j], _] = getSSE(x, y, A[i, j], M[i, j])
            # BCE[i, j] = np.sum(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))
    return A, M, SSE


########################################################################################################################
def plotLossLandscape(x, y, a_lim, m_lim, resolution):
    [A, M, SSE] = getRaster(x, y, a_lim, m_lim, resolution)

    # # Create 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, M, SSE, cmap='jet', alpha=1, linewidth=0.1, edgecolor='black')
    # plt.contour(A, M, SSE, cmap='jet', levels=100)

    # Add labels and title
    ax.set_xlabel('Slope - a', fontsize=14)
    ax.set_ylabel('Intercept - m', fontsize=14)
    ax.set_zlabel('SSE - Loss', fontsize=14)
    ax.view_init(elev=25, azim=-20)
    plt.title('Loss Landscape in 3D', fontsize=16, fontweight='bold')
    plt.xlim(a_lim[0], a_lim[1])
    plt.ylim(m_lim[0], m_lim[1])
    plt.tight_layout()
    plt.show()


########################################################################################################################
def getPerformance(x, y, a, m, threshold):
    # Predict the classes
    y_pred = 1/(1 + np.exp(-(a*x + m))) >= threshold
    # Calculate confusion matrix elements
    TP = sum((y_pred == 1) & (y == 1))  # True Positives
    TN = sum((y_pred == 0) & (y == 0))  # True Negatives
    FP = sum((y_pred == 1) & (y == 0))  # False Positives
    FN = sum((y_pred == 0) & (y == 1))  # False Negatives

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    sensitivity = TP / (TP + FN) * 100
    specificity = TN / (TN + FP) * 100

    print("\nClassification performance:\nAccuracy:    {:.2f}%".format(accuracy))
    print("Sensitivity: {:.2f}%".format(sensitivity))
    print("Specificity: {:.2f}%\n".format(specificity))







