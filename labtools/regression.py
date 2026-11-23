import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#title Funzione del fit lineare (iterabile per incertezze sulle x)
#title  fit di Santanastasio

def my_mean(x, w):
    return np.sum( x*w ) / np.sum( w )

def my_cov(x, y, w):
    return my_mean(x*y, w) - my_mean(x, w)*my_mean(y, w)

def my_var(x, w):
    return my_cov(x, x, w)

def my_line(x, m=1, c=0):
    return m*x + c

def y_estrapolato(x, m, c, sigma_m, sigma_c, cov_mc):
    y = m*x + c
    uy = np.sqrt((x**2)*(sigma_m**2)+
                   sigma_c**2 + 2*x*cov_mc)
    return y, uy #ho propagato semplicemente l'incertezza sulla y avendo assunto x noto con precisione massima

def lin_fit_0(x, y, sd_y, xlabel="x [ux]", ylabel="y [uy]", xm=0., xM=1., ym=0., yM=1., verbose=True, plot=False, setrange=False):
    #pesi
    w_y = np.power(sd_y.astype(float), -2)

    #m
    m = my_cov(x, y, w_y) / my_var(x, w_y)
    var_m = 1 / ( my_var(x, w_y) * np.sum(w_y) )

    #c
    c = my_mean(y, w_y) - my_mean(x, w_y) * m
    var_c = my_mean(x**2, w_y) * var_m

    #cov
    cov_mc = - my_mean(x, w_y) * var_m

    #rho
    rho_mc = cov_mc / ( np.sqrt(var_m) * np.sqrt(var_c) )

    if (verbose):

        print ('m         = ', m.round(8))
        print ('sigma(m)  = ', np.sqrt(var_m).round(8))
        print ('c         = ', c.round(8))
        print ('sigma(c)  = ', np.sqrt(var_c).round(8))
        print ('cov(m, c) = ', cov_mc.round(8))
        print ('rho(m, c) = ', rho_mc.round(8))

    if (plot):

        # rappresento i dati
        plt.errorbar(x, y, yerr=sd_y, xerr=0, ls='', marker='.',
                     color="black",label='dati')

        # costruisco dei punti x su cui valutare la retta del fit
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        xmin_plot = max(-0.0005, xmin-.2*(xmax-xmin))
        xmax_plot = xmax+.2*(xmax-xmin)
        if (setrange):
            xmin_plot = xm
            xmax_plot = xM
        x1 = np.linspace(xmin_plot, xmax_plot, 100)
        y1 = my_line(x1, m, c)

        # rappresento la retta del fit
        plt.plot(x1, y1, linestyle='-.', color="green", label='fit')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Fit lineare')
        plt.xlim(xmin_plot,xmax_plot)
        if (setrange):
            plt.ylim(ym,yM)

        # rappresento le incertezze sulla retta
        y1_plus_1sigma = y1+y_estrapolato(x1, m, c, np.sqrt(var_m), np.sqrt(var_c), cov_mc)[1]
        y1_minus_1sigma = y1-y_estrapolato(x1, m, c, np.sqrt(var_m), np.sqrt(var_c), cov_mc)[1]
        plt.plot(x1,y1_plus_1sigma, linestyle='-', color="orange", label=r'fit $\pm 1\sigma$')
        plt.plot(x1,y1_minus_1sigma, linestyle='-', color="orange")

        plt.grid(linestyle='-',color='#D5D1CD',alpha=.4)


        plt.legend()

    results = np.array([my_mean(x, w_y),
                        my_mean(y, w_y),
                        my_mean(x**2, w_y),
                        my_mean(x*y, w_y),
                        my_var(x, w_y),
                        my_cov(x,y, w_y),
                        np.sum(w_y)])

    return m, np.sqrt(var_m), c, np.sqrt(var_c), cov_mc, rho_mc, results

################################################################
#fit iterato (1 volta) con incertezze sulle x

def lin_fit(x, ux, y, uy, xlabel="x [ux]", ylabel="y [uy]", xm=0., xM=1., ym=0., yM=1., verbose=True, plot=True, setrange=False):

    m0, sm0, c0, sc0, cov0, rho0, results_0 = lin_fit_0(x, y, uy, verbose=False, plot=False, setrange=False)

    #u_y nuove, tenendo conto anche delle u_x
    uy_new = np.sqrt(uy**2+(m0*ux)**2)

    m, sm, c, sc, cov, rho, results = lin_fit_0(x, y, uy_new, xlabel, ylabel, xm, xM, ym, yM,
            verbose, plot, setrange)

    return m, sm, c, sc, cov, rho, results
