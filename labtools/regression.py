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

        # Rappresento i dati sperimentali selezionando un sottoinsieme
        step = max(1, len(x) // 200)

        # Creo gli indici dei punti da rappresentare
        idx = np.arange(0, len(x), step)

        # Mi assicuro di includere anche l'ultimo punto
        if idx[-1] != len(x) - 1:
            idx = np.append(idx, len(x) - 1)

        plt.errorbar(
            x[idx], y[idx],
            yerr=sd_y[idx],
            xerr=0,
            ls='',
            marker='.',
            color="black",
            label='dati'
        )

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
        plt.xlim(xmin_plot, xmax_plot)

        if (setrange):
            plt.ylim(ym, yM)

        # rappresento le incertezze sulla retta
        y1_plus_1sigma = y1 + y_estrapolato(
            x1, m, c, np.sqrt(var_m), np.sqrt(var_c), cov_mc
        )[1]

        y1_minus_1sigma = y1 - y_estrapolato(
            x1, m, c, np.sqrt(var_m), np.sqrt(var_c), cov_mc
        )[1]

        plt.plot(x1, y1_plus_1sigma, linestyle='-', color="orange",
                 label=r'fit $\pm 1\sigma$')
        plt.plot(x1, y1_minus_1sigma, linestyle='-', color="orange")

        plt.grid(linestyle='-', color='#D5D1CD', alpha=.4)
        plt.legend()

    results = np.array([
        my_mean(x, w_y),
        my_mean(y, w_y),
        my_mean(x**2, w_y),
        my_mean(x*y, w_y),
        my_var(x, w_y),
        my_cov(x, y, w_y),
        np.sum(w_y)
    ])

    return m, np.sqrt(var_m), c, np.sqrt(var_c), cov_mc, rho_mc, results








################################################################
#fit iterato (nmax volte, di default è settata a 1) con incertezze sulle x

def lin_fit_iter(x, ux, y, uy, n_max=1, tol=1e-8, verbose=True, plot=False,
                 xlabel="x [ux]", ylabel="y [uy]", xm=0., xM=1., ym=0., yM=1., setrange=False):
    """
    Fit lineare con incertezze su x e y, iterato fino a convergenza.

    Procedimento
    ------------
    1. Primo fit con lin_fit_0 usando solo uy  ->  m0.
    2. Ripete: uy_eff = sqrt(uy**2 + (m*ux)**2), poi nuovo fit con lin_fit_0  ->  nuovo m.
    3. Si ferma quando |m_nuovo - m_vecchio| <= tol*|m_nuovo|, oppure dopo n_max iterazioni.

    Con n_max=1 equivale a lin_fit (un solo aggiornamento).

    Parameters
    ----------
    x, ux, y, uy : array_like o scalari
        Dati e incertezze assolute (ux puo' essere 0).
    n_max : int
        Numero massimo di aggiornamenti delle incertezze. Default: 1.
    tol : float
        Soglia di convergenza RELATIVA sulla pendenza. Default: 1e-8.
    verbose, plot, xlabel, ylabel, xm, xM, ym, yM, setrange :
        Come in lin_fit_0 (riferiti al fit finale).

    Returns
    -------
    m, sm, c, sc, cov, rho, results :
        Come lin_fit_0 (fit finale).
    uy_eff : ndarray
        Incertezze sulle y effettivamente usate nel fit finale (servono per il chi2).
    n_iter : int
        Numero di aggiornamenti eseguiti.
    """
    # Converto in array (gli scalari diventano array lunghi quanto x)
    x  = np.asarray(x,  dtype=float)
    y  = np.asarray(y,  dtype=float)
    ux = np.asarray(ux, dtype=float) * np.ones_like(x)
    uy = np.asarray(uy, dtype=float) * np.ones_like(x)

    # Primo fit: solo incertezze sulle y
    m_old = lin_fit_0(x, y, uy, verbose=False)[0]

    converged = False
    for n_iter in range(1, n_max + 1):
        uy_eff = np.sqrt(uy**2 + (m_old * ux)**2)          # incertezze effettive con il m corrente
        m_new = lin_fit_0(x, y, uy_eff, verbose=False)[0]  # nuovo fit
        converged = abs(m_new - m_old) <= tol * abs(m_new)
        m_old = m_new
        if converged:
            break

    if not converged:
        print(f"ATTENZIONE: nessuna convergenza dopo {n_max} iterazioni (tol = {tol}).")

    # Fit finale con le incertezze uy_eff (stampa/grafico opzionali)
    m, sm, c, sc, cov, rho, results = lin_fit_0(x, y, uy_eff, xlabel, ylabel, xm, xM, ym, yM,
                                                verbose, plot, setrange)
    if verbose:
        print('iterazioni =', n_iter)

    return m, sm, c, sc, cov, rho, results, uy_eff, n_iter

#Chi2_lin
def chi2_lin(x, y, uy, m, c, ridotto=False, n_par=2,
             verbose=True, plot=False,
             xlabel="x", ylabel="Residui"):

    """
    Calcola il chi quadro di una retta y = m*x + c.

    Se plot=True, rappresenta i residui rispetto a x.
    """

    x  = np.asarray(x, dtype=float)
    y  = np.asarray(y, dtype=float)
    uy = np.asarray(uy, dtype=float) * np.ones_like(x)

    if np.any(uy <= 0):
        raise ValueError("Le incertezze devono essere positive.")

    if x.shape != y.shape or x.shape != uy.shape:
        raise ValueError("x, y e uy devono avere la stessa lunghezza.")

    ndf = len(x) - n_par

    if ndf <= 0:
        raise ValueError("ndf <= 0: servono piu' punti che parametri.")

    # Calcolo dei residui
    residui = y - (m*x + c)

    # Chi quadro
    chi2 = np.sum((residui / uy)**2)
    valore = chi2 / ndf if ridotto else chi2

    if verbose:
        print('N         = ', len(x))
        print('ndf       = ', ndf)
        print('chi2      = ', round(chi2, 4))
        print('chi2/ndf  = ', round(chi2 / ndf, 4))

    # Grafico dei residui
    if plot:

        plt.figure(figsize=(8, 5))

        # Rappresento un sottoinsieme dei residui per evitare sovrapposizioni
        step = max(1, len(x) // 200)
        idx = np.arange(0, len(x), step)

        # Mi assicuro di includere anche l'ultimo punto
        if idx[-1] != len(x) - 1:
            idx = np.append(idx, len(x) - 1)

        plt.errorbar(
            x[idx], residui[idx],
            yerr=uy[idx],
            fmt='o',
            color='black',
            capsize=3,
            label='Residui'
        )

        plt.axhline(
            0,
            color='red',
            linestyle='--',
            label='Residuo nullo'
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Residui del fit lineare')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

    return valore, ndf
