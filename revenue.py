import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from mpl_toolkits import mplot3d
import nlopt
from scipy.optimize import minimize
from scipy.optimize import basinhopping

np.random.seed(100)
N = 5000
r = 0.05
dt = 1/48
T = 2
rho = 0.3
galpha = 3
calpha = 7
jalpha = 20
gtheta = [0 for i in range(12)]
ctheta = [0 for i in range(12)]
gsigma = 0.0722*np.sqrt(1/dt)
csigma1 = 0.2887*np.sqrt(1/dt)
csigma2 = 0.1083*np.sqrt(1/dt)
lamdadt = 0.083
m = 75
g0 = 3
c0 = 35
j0 = 0
r = 0.05

steps = int(T/dt)
Zp = np.random.normal(size = (steps, N))
e = np.random.normal(size = (steps, N))
Zg = rho * Zp + np.sqrt(1-rho**2) * e

# gas process
def gt(alpha, theta, sigma, Z, t, g0):
    steps, N = Z.shape
    steps = int(t/dt)
    g = np.zeros((steps+1, N))
    g[0] = g0
    for i in range(steps):
        g[i+1] = g[i] + alpha *(theta[(i%48)//4]-g[i])*dt + sigma*g[i]*np.sqrt(dt)*Z[i]
    return g


# power process
def pt(alpha1, alpha2, theta, sigma1, sigma2, Z, t, c0, j0, lamdadt, m):
    steps, N = Z.shape
    steps = int(t/dt)
    c = np.zeros((steps+1, N))
    c[0] = c0
    j = np.zeros((steps + 1, N))
    j[0] = j0
    for i in range(steps):
        if 4 <= (i%48)//4 <= 7:
            sigma = sigma1
        else:
            sigma = sigma2
        c[i+1] = c[i] + alpha1 *(theta[(i%48)//4]-c[i])*dt + sigma*c[i]*np.sqrt(dt)*Z[i]
        j[i+1] = j[i] + alpha2 * (0 - j[i]) * dt
        unif = np.random.uniform(size=N)
        j[i+1] += (c[i] + j[i] > 75) * (unif < lamdadt) * m
    return c+j


gtheta = [2.4633166603436436, 2.4633166603436436, 2.6988247470228335, 2.5598825281725417, \
          5.9854972185418704, 6.260238307153297, 6.490037007222248, 5.703977540876591, \
          2.621531073688348, 2.6527080824715092, 1.8632108979561492, 3.4029831126259045]

ctheta = [39.2016363542152, 39.2016363542152, 40.367874102778096, 42.33623080036156, \
          43.23, 58.38189861966566, 72.7, 85.08519562168547, \
          76.96221282403006, 61.32, 50.75, 46.9]

g = gt(galpha, gtheta, gsigma, Zg, T, g0)
p = pt(calpha, jalpha, ctheta, csigma1, csigma2, Zp, T, c0, j0, lamdadt, m)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    g_mean = [np.mean(row) for row in g]
    plt.plot(g_mean)
    plt.title("Gas Price Mean Over Two Years")
    plt.xlabel("Week")
    plt.ylabel("Gas Price")
    plt.show()

    g_std = [np.std(row) for row in g]
    plt.plot(g_std)
    plt.title("Gas Price Standard Deviation Over Two Years")
    plt.xlabel("Week")
    plt.ylabel("Gas Price Deviation")
    plt.show()

    p_mean = [np.mean(row) for row in p]
    plt.plot(p_mean)
    plt.title("Power Price Mean Over Two Years")
    plt.xlabel("Week")
    plt.ylabel("Power Price")
    plt.show()

    p_std = [np.std(row) for row in p]
    plt.plot(p_std)
    plt.title("Power Price Standard Deviation Over Two Years")
    plt.xlabel("Week")
    plt.ylabel("Power Price Deviation")
    plt.show()

    spread = p - 12*g - 5
    spread_mean = [np.mean(row) for row in spread]
    plt.plot(spread_mean)
    plt.title("Spark Spread Mean Over Two Years")
    plt.xlabel("Week")
    plt.ylabel("Price Spread")
    plt.show()

    spread_std = [np.std(row) for row in spread]
    plt.plot(spread_std)
    plt.title("Spark Spread Standard Deviation Over Two Years")
    plt.xlabel("Week")
    plt.ylabel("Price Spread")
    plt.show()



# # 0: close; 1:open
# status = np.zeros((48*T + 1, N))
# status[0] = 0


# def objective(param):
#     cto = param[0]
#     otc = param[1]
#     revenue = 0
#     profit = np.zeros((12*T, N)) # revue delivered monthly
#     for i in range(steps):
#         spread = p[i] - 12*g[i] - 5
#         # if closed i, check if i+1 should be open or not
#         status[i + 1] = np.logical_and((status[i] == 0), (spread > cto)) ##
#         # if open in i, check if i+1 should be closed or not
#         status[i + 1] += np.logical_and((status[i] == 1), (spread < otc))
#         # if close, but will open in next period
#         profit[i//4][np.logical_and(status[i] == 0, status[i+1] == 1)] += -3*1000*16*7.6
#         # if open
#         profit[i//4][status[i] == 1] += spread[status[i] == 1]*1000*16*7.6
#     for j in range(12*T):
#         revenue += np.mean(profit[j]) * np.exp(-r * (j+1)/12)
#     return -revenue


# cto = 0
# otc = 0
# x0 = (cto, otc)
# #bnds = ((0, np.max(p)), (None, np.max(p))) ##
# bnds = ((None, None), (None, None)) ##
# options={'disp':False, 'maxiter':1000, 'ftol':1e-20}
# res = basinhopping(objective, x0=x0)
# print(res)


'''
def operate(param):
    cto = param[0]
    otc = param[1]
    revenue = 0
    profit = np.zeros((12*T, N)) # revue delivered monthly
    for i in range(steps):
        spread = p[i] - 12*g[i] - 5
        # if closed i, check if i+1 should be open or not
        status[i + 1] = (status[i] == 0) * (spread > cto)
        # if open in i, check if i+1 should be closed or not
        status[i + 1] += (status[i] == 1) * (spread < otc)
        # if close, but will open in next period
        profit[i//4][np.logical_and(status[i] == 0, status[i+1] == 1)] += -3*1000*16*7.6
        # if open
        profit[i//4][status[i] == 1] += spread[status[i] == 1]*1000*16*7.6
    for j in range(12*T):
        revenue += np.mean(profit[j]) * np.exp(-r * (j+1)/12)
    return revenue

param = [8.06956081, -7.27546325]
'''
