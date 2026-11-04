# MSc MLS – CO3 Optimization: Logistic Regression on Iris

Master course project (CO3, Optimization).

A 1-D logistic regression classifies *Iris versicolor* vs. *Iris virginica* by petal length, and its two parameters (slope `a`, intercept `m`) are fitted with different optimization algorithms. 

The focus lies on building intuition for the inner workings of gradient decent and simulated annealing. The loss landscape and the optimizer path are visualized in image and video.

Model: `y_pred = 1 / (1 + exp(-(a*x + m)))`, where `x` is the mean-centred petal length.
Loss: sum of squared errors (SSE) or log loss (binary cross-entropy).

## Files

| File | Description |
|---|---|
| `Iris_LogReg_OptimizerVisualizer.m` | MATLAB: iterative optimization with live plots of loss landscape, parameter path and model fit (optional video export) |
| `FunLibLogReg.py` | Python function library: SSE loss, gradient descent, simulated annealing, SciPy `minimize` reference, loss-landscape plotting, performance metrics |
| `Gradient_Descent_LogReg.py` | Python: runs gradient descent with a sigmoid-shaped learning-rate schedule |
| `Simulated_Annealing_LogReg.py` | Python: runs simulated annealing with sigmoid-shaped step-size and temperature schedules |
| `iris_data.txt` | Iris dataset (JSON), read by the MATLAB scripts |
| `Final_Project_Report_CO3_ Basil_Imoberdorf.pdf` / `Final_Report_CO3_Basil_Imoberdorf.docx` | Project report |
| `Optimization_Project_CO3_LogReg_Basil_Imoberdorf.pptx` | Presentation slides |

## Usage

**MATLAB:** run `Iris_LogReg_Optimizer.m` from this folder (it reads `iris_data.txt`). Settings at the top of the script: initial guess (`a_0`, `b_0`), loss (`ObjFun = 'SSE'` or `'LGL'`), number of iterations and `MakeMovie`.

**Python:** requires `numpy`, `scipy`, `matplotlib`, `scikit-learn` (the Iris data is loaded via `sklearn.datasets.load_iris`).

```bash
python Gradient_Descent_LogReg.py
python Simulated_Annealing_LogReg.py
```
