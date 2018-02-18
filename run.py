import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

from cf import option_price
from cn import CN_price
from backtest import backtest_strike, backtest_m


###############################   Parameters   ################################

CP = 1
S_0 = 100.
K = 100.
T = 1.
S_min = 0.
S_max = 200.
sigma = 0.2
r = 0.
N = 50
M = 50

#boudary condition at maturity: D for Dirichlet, N for Neumann
bc = 'D'
#type of grid, U for uniform, NU for non uniform
gt = 'U'


############################   Closed Form Method   ###########################
[cf_time, cf_price] = option_price(CP, S_0, K, r, sigma, T)

print("--Closed Form method")
print("Asset Price: ", S_0)
print("Call Option Price: ", cf_price)
print("Computation time", cf_time)


##########################   Crank-Nicholson Scheme   #########################
[V_S, V_time, UU,  V_price, cn_price, cn_time] = CN_price(T, K, S_0, r, sigma, S_min, S_max, N, M, bc, gt)
print("\n--Crank-Nicholson method")
print("Asset Price: ", S_0)
print("Call Option Price: ", cn_price)
print("Error: ", abs(cn_price - cf_price))
print("Computation time", cn_time)

#plot 2D V_price against S
ds = (S_max - S_min) / float(M)
plt.title('Call prices against the stock price at time t=0')
plt.xlabel('Underlying price'); plt.ylabel('Call price')
plt.plot(V_S, V_price)
plt.plot(V_S, np.array([max(V_S[j] - K, 0) for j in range(M+1)]))
plt.show()

UU = np.array(UU)


################   Convergence as K varies in a fixed grid   ##################
#Set of strikes
l_K = [50, 75, 100, 125, 150]

#what we want to compare:
bc1 = 'D'
bc2 = 'N'

gt1 = 'U'
gt2 = 'U'

#mesh accuracy
l_n = [50]

backtest_strike(CP, S_0, r, sigma, T, l_K, l_n, bc1, bc2, gt1, gt2)

########################   Convergence as m1 varies   #########################
#We do that for these 3 cases
l_K = [80, 100, 120]

#what we want to compare:
bc1 = 'D'
bc2 = 'N'

gt1 = 'U'
gt2 = 'U'

#mesh accuracy
l_n = [50, 75, 100]

backtest_m(CP, S_0, r, sigma, T, l_K, l_n, bc1, bc2, gt1, gt2)
