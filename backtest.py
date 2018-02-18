from cf import option_price
from cn import CN_price
import matplotlib.pylab as plt


def backtest_strike(CP, S_0, r, sigma, T, l_K, l_n, bc1, bc2, gt1, gt2):
    m = 20
    n = 50

    plt.xlabel('Strike')
    plt.ylabel('Relative Error (%)')

    V_err_r_1 = []
    V_err_r_2 = []  # relative error with Neumann boundary condition

    for K in l_K:
        [time_cf, price_cf] = option_price(1, S_0, K, r, sigma, T)
        [V_S_1, V_time_1, UU_1, V_price_1, price_cn_1, time_cn_1] = CN_price(T, K, S_0, r, sigma, 0, 300, n, m, bc1, gt1)
        [V_S_2, V_time_2, UU_2, V_price_2, price_cn_2, time_cn_2] = CN_price(T, K, S_0, r, sigma, 0, 300, n, m, bc2, gt2)
        V_err_r_1.append(abs(price_cf - price_cn_1)/price_cf)
        V_err_r_2.append(abs(price_cf - price_cn_2) / price_cf)

    plt.plot(l_K, V_err_r_1, label='{}; {}'.format(bc1, gt1))
    plt.plot(l_K, V_err_r_2, label='{}; {}'.format(bc2, gt2))
    plt.legend(loc='upper left', shadow = True)

    plt.show()
    return


def backtest_m(CP, S_0, r, sigma, T, l_K, l_n, bc1, bc2, gt1, gt2):
    #list containing several meshing, from 3 to 41
    l_m = [2*i + 1 for i in range(1, 21)]

    for n in l_n:
        # one plot for each n
        fig = plt.figure()

        ax1 = fig.add_subplot(211)
        ax1.set_ylabel('Relative Error (%)')
        ax1.set_title('n = {}'.format(n))

        ax2 = fig.add_subplot(212)
        ax2.set_xlabel('number of points along S-axis')
        ax2.set_ylabel('Time (seconds)')


        for K in l_K:
            [time_cf, price_cf] = option_price(1, S_0, K, r, sigma, T)
            line_data = [time_cf for i in range(len(l_m))]  #used to plot the constant computation time of closed form

            V_err_r_1 = []
            V_err_r_2 = []  # relative error with Neumann boundary condition
            V_tcn_1 = []    # computation time of CN with D
            V_tcn_2 = []    # computation time of CN with N
            for m in l_m:
                [V_S_1, V_time_1, UU_1, V_price_1, price_cn_1, time_cn_1] = CN_price(T, K, S_0, r, sigma, K - 50, K + 50, n, m, bc1, gt1)
                [V_S_2, V_time_2, UU_2, V_price_2, price_cn_2, time_cn_2] = CN_price(T, K, S_0, r, sigma, K - 50, K + 50, n, m, bc2, gt2)
                V_err_r_1.append(abs(price_cf - price_cn_1)/price_cf)
                V_err_r_2.append(abs(price_cf - price_cn_2) / price_cf)
                V_tcn_1.append(time_cn_1)
                V_tcn_2.append(time_cn_2)

            ax1.plot(l_m, V_err_r_1, label='{}; {}; K = {}'.format(bc1, gt1, K))
            ax1.plot(l_m, V_err_r_2, linestyle=':', label='{}; {}; K = {}'.format(bc2, gt2, K))
            ax1.legend(loc='upper right', shadow=True)

            ax2.plot(l_m, V_tcn_1, label='{}; {}; K = {}'.format(bc1, gt1, K))
            ax2.plot(l_m, V_tcn_2, linestyle=':', label='{}; {}; K = {}'.format(bc2, gt2, K))
            ax2.legend(loc='upper left', shadow=True)
            plt.show()
    return
