import numpy as np
import matplotlib.pyplot as plt

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
    """
    Esegue un fit lineare pesato considerando le incertezze sulle ordinate
    e assumendo trascurabili le incertezze sulle ascisse.

    Il modello utilizzato è:
        y = m*x + c

    Il fit viene effettuato mediante il metodo dei minimi quadrati pesati,
    con pesi:
        w_i = 1 / sd_y_i**2

    Parameters
    ----------
    x : array_like
        Valori sperimentali delle ascisse.

    y : array_like
        Valori sperimentali delle ordinate.

    sd_y : array_like
        Incertezze assolute sui valori di y.
        Devono essere strettamente positive.

    xlabel : str, optional
        Etichetta dell'asse x del grafico.
        Default: "x [ux]".

    ylabel : str, optional
        Etichetta dell'asse y del grafico.
        Default: "y [uy]".

    xm, xM : float, optional
        Limite inferiore e superiore dell'asse x.
        Vengono utilizzati soltanto se setrange=True.

    ym, yM : float, optional
        Limite inferiore e superiore dell'asse y.
        Vengono utilizzati soltanto se setrange=True.

    verbose : bool, optional
        Se True, stampa i risultati numerici del fit.
        Default: True.

    plot : bool, optional
        Se True, rappresenta i dati sperimentali, la retta di fit
        e le bande di incertezza a 1 sigma.
        Default: False.

    setrange : bool, optional
        Se True, utilizza i limiti degli assi specificati
        attraverso xm, xM, ym e yM.
        Default: False.

    Returns
    -------
    m : float
        Coefficiente angolare della retta.

    sm : float
        Incertezza sul coefficiente angolare.

    c : float
        Intercetta della retta.

    sc : float
        Incertezza sull'intercetta.

    cov_mc : float
        Covarianza tra coefficiente angolare e intercetta.

    rho_mc : float
        Coefficiente di correlazione tra m e c.

    results : ndarray
        Array contenente, nell'ordine:
        [media pesata di x,
         media pesata di y,
         media pesata di x**2,
         media pesata di x*y,
         varianza pesata di x,
         covarianza pesata tra x e y,
         somma dei pesi].

    Notes
    -----
    Le incertezze sulle ascisse non vengono considerate.
    Per tenerne conto utilizzare lin_fit().

    L'incertezza sulla retta estrapolata viene calcolata
    assumendo x noto con precisione massima.

    Examples
    --------
    >>> m, sm, c, sc, cov, rho, results = lin_fit_0(x, y, uy)
    """
    
    #Converto in array numpy
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sd_y = np.asarray(sd_y, dtype=float)
    
    #Controllo che le incertezze sulle y siano positive
    if np.any(sd_y <= 0):
    raise ValueError("Le incertezze devono essere positive.")
    
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
        xmin_plot = xmin-.2*(xmax-xmin)
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
    """
    Esegue un fit lineare considerando le incertezze sia sulle ascisse
    sia sulle ordinate, mediante una procedura iterativa a un solo
    aggiornamento delle incertezze.

    Il modello utilizzato è:
        y = m*x + c

    PROCEDIMENTO
    -----------
    1. Esegue un primo fit mediante lin_fit_0(), considerando
       esclusivamente le incertezze sulle ordinate uy.

    2. Utilizza il coefficiente angolare m0 ottenuto per calcolare
       le nuove incertezze effettive sulle ordinate:

           uy_new = sqrt(uy**2 + (m0*ux)**2)

    3. Esegue un secondo fit mediante lin_fit_0(), utilizzando
       le nuove incertezze uy_new.

    Il risultato restituito è quello del secondo fit.

    Parameters
    ----------
    x : array_like
        Valori sperimentali delle ascisse.

    ux : array_like
        Incertezze assolute sui valori di x.
        Devono essere non negative.

    y : array_like
        Valori sperimentali delle ordinate.

    uy : array_like
        Incertezze assolute sui valori di y.
        Devono essere strettamente positive.

    xlabel : str, optional
        Etichetta dell'asse x del grafico.
        Default: "x [ux]".

    ylabel : str, optional
        Etichetta dell'asse y del grafico.
        Default: "y [uy]".

    xm, xM : float, optional
        Limite inferiore e superiore dell'asse x.
        Vengono utilizzati soltanto se setrange=True.

    ym, yM : float, optional
        Limite inferiore e superiore dell'asse y.
        Vengono utilizzati soltanto se setrange=True.

    verbose : bool, optional
        Se True, stampa i risultati numerici del secondo fit.
        Default: True.

    plot : bool, optional
        Se True, rappresenta i dati sperimentali, la retta di fit
        e le bande di incertezza a 1 sigma.
        Default: True.

    setrange : bool, optional
        Se True, utilizza i limiti degli assi specificati
        attraverso xm, xM, ym e yM.
        Default: False.

    Returns
    -------
    m : float
        Coefficiente angolare del secondo fit.

    sm : float
        Incertezza sul coefficiente angolare.

    c : float
        Intercetta del secondo fit.

    sc : float
        Incertezza sull'intercetta.

    cov : float
        Covarianza tra coefficiente angolare e intercetta.

    rho : float
        Coefficiente di correlazione tra m e c.

    results : ndarray
        Array contenente le quantità statistiche del secondo fit,
        con la stessa struttura restituita da lin_fit_0().

    Notes
    -----
    La procedura effettua un solo aggiornamento delle incertezze:
    non è quindi un algoritmo iterativo fino a convergenza e non
    costituisce un fit ODR completo.

    Si assume che le incertezze sulle x e sulle y siano indipendenti.

    Examples
    --------
    >>> m, sm, c, sc, cov, rho, results = lin_fit(x, ux, y, uy)
    """

    #Converto x, ux, y, uy in array numpy
    x = np.asarray(x, dtype=float)
    ux = np.asarray(ux, dtype=float)
    y = np.asarray(y, dtype=float)
    uy = np.asarray(uy, dtype=float)

    #Effettuo una prima iterazione mediante lin_fit_0
    m0, sm0, c0, sc0, cov0, rho0, results_0 = lin_fit_0(x, y, uy, verbose=False, plot=False, setrange=False)
    
    #u_y nuove, tenendo conto anche delle u_x
    uy_new = np.sqrt(uy**2+(m0*ux)**2)

    m, sm, c, sc, cov, rho, results = lin_fit_0(x, y, uy_new, xlabel, ylabel, xm, xM, ym, yM,
            verbose, plot, setrange)

    return m, sm, c, sc, cov, rho, results
