import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
# from uncertainties.umath import *

# KONSTANTEN
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
p_0 = 1013.25  # mbar
T_0 = 273.15 + 15  # °C
A = 1e2
R = ufloat(8.3144598, 0.0000048)  # J mol-1 K-1
T = 273.15 + 24.8  # °C
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

######################################################################
# calculate contrast from voltage measurments for different
# angles of the polarisation filter. fit of a sin(2x)².
######################################################################

winkel, U_min, U_max = np.genfromtxt('kontrast.txt', unpack='True')


def contrast(max, min):
    return (max-min)/(max+min)


def fit(a, x):
    return a*(np.sin(2*x))**2


params, cov = curve_fit(fit, np.radians(winkel), contrast(U_max, U_min), bounds=(0.8, 0.85))
a = params[0]
a_err = np.sqrt(cov[0][0])

x = np.linspace(0, 180, 10000)
plt.plot(winkel, contrast(U_max, U_min), 'rx', label='Messwerte')
plt.plot(x, fit(a, np.radians(x)), label=r'$A\cdot \sin(2x)^2$, A = {:.3f}$\pm${:.3f}'.format(a,a_err))
plt.xlabel(r'Winkel $\phi$ in °')
plt.ylabel('Kontrast')
plt.ylim(0, 1)
plt.title('Winkelverteilung des Kontrastes')
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('kontrast.pdf')
plt.close()
print('''Maxima des Kontrastes nach a sin(2x)² bei den Winkeln 45° und 135°.
''')

######################################################################
# calculate refraction index of the glass slobe from interference
# counts for different angles of the glass slobe (10° steps).
######################################################################

number, counts = np.genfromtxt('glas.txt', unpack='True')
phi_1 = number*10
phi_1 = np.radians(phi_1)
phi_2 = phi_1 + np.radians(10)

vac_wavelen = 632.99e-9
thickness = 0.5e-3
# ref_index = 1/(1 - counts * vac_wavelen/(thickness * (phi_2**2 - phi_1**2)))
alpha = counts*vac_wavelen/(2*thickness)
ref_index = (alpha**2+2*(1-np.cos(phi_1))*(1-alpha))/(2*(1-np.cos(phi_1)-alpha))

print('''
Refraction index of glass slabs:   {:.4f} ± {:.4f}'''
      .format(np.mean(ref_index), ref_index.std(ddof=1)))

######################################################################
# calculate refraction index of CO2 chamber from interference
# counts for different pressures of mentioned gas. (50mbar steps)
######################################################################

pressure, counts1, counts2, counts3 = np.genfromtxt('CO2.txt', unpack='True')
xplotlin = np.linspace(pressure[0], pressure[-1], 2)
length = ufloat(100e-3, 0.1e-3)

ref_index1 = counts1 * vac_wavelen / length + 1
ref_index2 = counts2 * vac_wavelen / length + 1
ref_index3 = counts3 * vac_wavelen / length + 1

print('''
Refraction index of CO2:   {:.5f} for (24.7°, 18mbar)
Refraction index of CO2:   {:.5f} for (24.8°, 4mbar)
Refraction index of CO2:   {:.5f} for (24.8, 3mbar)'''.format(np.mean(ref_index1), np.mean(ref_index2), np.mean(ref_index3)))
ref_index1_err = np.zeros(len(ref_index1))
ref_index2_err = np.zeros(len(ref_index1))
ref_index3_err = np.zeros(len(ref_index1))
refindex1 = np.zeros(len(ref_index1))
refindex2 = np.zeros(len(ref_index1))
refindex3 = np.zeros(len(ref_index1))

for i in range(0, len(ref_index1)):
    ref_index1_err[i] = stds(ref_index1[i])
    refindex1[i] = noms(ref_index1[i])

for i in range(0, len(ref_index2)):
    ref_index2_err[i] = stds(ref_index2[i])
    refindex2[i] = noms(ref_index2[i])

for i in range(0, len(ref_index3)):
    ref_index3_err[i] = stds(ref_index3[i])
    refindex3[i] = noms(ref_index3[i])


def lin(a, b, x):
    return a*x+b


fit1_params, fit1_cov = np.polyfit(pressure, refindex1, 1, cov=True)
errors1 = np.sqrt(np.diag(fit1_cov))

fit2_params, fit2_cov = np.polyfit(pressure, refindex2, 1, cov=True)
errors2 = np.sqrt(np.diag(fit2_cov))

fit3_params, fit3_cov = np.polyfit(pressure, refindex3, 1, cov=True)
errors3 = np.sqrt(np.diag(fit3_cov))

steig = np.array([fit1_params[0], fit2_params[0], fit3_params[0]])
achse = np.array([fit1_params[1], fit2_params[1], fit3_params[1]])

print('''
Ergebnisse Ausgleichsrechnung CO2:
Messung 1: a = {} ± {}   b = {} ± {}
Messung 2: a = {} ± {}   b = {} ± {}
Messung 3: a = {} ± {}   b = {} ± {}
Mittwelwert: a = {} ± {}   b = {} ± {}
'''.format(fit1_params[0], errors1[0], 1-fit1_params[1], errors1[1], fit2_params[0], errors2[0], 1-fit2_params[1], errors2[1], fit3_params[0], errors3[0], 1-fit3_params[1], errors3[1], np.mean(steig), np.std(steig), 1-np.mean(achse), np.std(achse)))

m = ufloat(np.mean(steig), np.std(steig))
b = ufloat(np.mean(achse), np.std(achse))

n_norm = b + m * p_0 * T / T_0 # m*p_0*((3*A)/(R*T_0))+b

print('''
Berechneter Brechungsindex bei 1018hPa:
CO2: n = {} ± {}
Berechneter Brechungsindex bei Normalbedingungen:
CO2: n = {}
'''.format(np.mean(steig)*1018+np.mean(achse), np.std(achse)+np.std(steig), n_norm))

x = np.linspace(0, 1000, 1000)
plt.errorbar(pressure, refindex1, 100*ref_index1_err, fmt='x', label=r'Messung 1 (24.7°, 18mbar) Fehler 100fach vergrößert.')
plt.plot(x, lin(*fit1_params, x), label='Ausgleichsgeraden')
plt.title(r'Druckverteilung des Brechungsindex von $CO_2$')
plt.xlabel('Druck p/mbar')
plt.ylabel('Brechungsindex n')
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('CO2_1.pdf')
plt.close()
plt.errorbar(pressure, refindex2, 100*ref_index2_err, fmt='x', label=r'Messung 2 (24.8°, 4mbar) Fehler 100fach vergrößert.')
plt.plot(x, lin(*fit2_params, x), label='Ausgleichsgeraden')
plt.title(r'Druckverteilung des Brechungsindex von $CO_2$')
plt.xlabel('Druck p/mbar')
plt.ylabel('Brechungsindex n')
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('CO2_2.pdf')
plt.close()
plt.errorbar(pressure, refindex3, 100*ref_index3_err, fmt='x', label=r'Messung 3 (24.8, 3mbar) Fehler 100fach vergrößert.')
plt.plot(x, lin(*fit3_params, x), label='Ausgleichsgeraden')
plt.title(r'Druckverteilung des Brechungsindex von $CO_2$')
plt.xlabel('Druck p/mbar')
plt.ylabel('Brechungsindex n')
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('CO2_3.pdf')
plt.close()

######################################################################
# calculate refraction index of air chamber from interference
# counts for different pressures of mentioned gas. (50mbar steps)
######################################################################

pressure, counts1, counts2, counts3 = np.genfromtxt('luft.txt', unpack='True')
length = ufloat(100e-3, 0.1e-3)

ref_index1 = counts1 * vac_wavelen / length + 1
ref_index2 = counts2 * vac_wavelen / length + 1
ref_index3 = counts3 * vac_wavelen / length + 1

print('''
Refraction index of air:   {:.5f}
Refraction index of air:   {:.5f}
Refraction index of air:   {:.5f} '''.format(np.mean(ref_index1), np.mean(ref_index2), np.mean(ref_index3)))
ref_index1_err = np.zeros(len(ref_index1))
ref_index2_err = np.zeros(len(ref_index1))
ref_index3_err = np.zeros(len(ref_index1))
refindex1 = np.zeros(len(ref_index1))
refindex2 = np.zeros(len(ref_index1))
refindex3 = np.zeros(len(ref_index1))

for i in range(0, len(ref_index1)):
    ref_index1_err[i] = stds(ref_index1[i])
    refindex1[i] = noms(ref_index1[i])

for i in range(0, len(ref_index2)):
    ref_index2_err[i] = stds(ref_index2[i])
    refindex2[i] = noms(ref_index2[i])

for i in range(0, len(ref_index3)):
    ref_index3_err[i] = stds(ref_index3[i])
    refindex3[i] = noms(ref_index3[i])

fit1_params, fit1_cov = np.polyfit(pressure, refindex1, 1, cov=True)
errors1 = np.sqrt(np.diag(fit1_cov))

fit2_params, fit2_cov = np.polyfit(pressure, refindex2, 1, cov=True)
errors2 = np.sqrt(np.diag(fit2_cov))

fit3_params, fit3_cov = np.polyfit(pressure, refindex3, 1, cov=True)
errors3 = np.sqrt(np.diag(fit3_cov))

steig = np.array([fit1_params[0], fit2_params[0], fit3_params[0]])
achse = np.array([fit1_params[1], fit2_params[1], fit3_params[1]])

print('''
Ergebnisse Ausgleichsrechnung Luft:
Messung 1: a = {} ± {}   b = {} ± {}
Messung 2: a = {} ± {}   b = {} ± {}
Messung 3: a = {} ± {}   b = {} ± {}
Mittwelwert: a = {} ± {}   b = {} ± {}
'''.format(fit1_params[0], errors1[0], 1-fit1_params[1], errors1[1], fit2_params[0], errors2[0], 1-fit2_params[1], errors2[1], fit3_params[0], errors3[0], 1-fit3_params[1], errors3[1], np.mean(steig), np.std(steig), 1-np.mean(achse), np.std(achse)))

m = ufloat(np.mean(steig), np.std(steig))
b = ufloat(np.mean(achse), np.std(achse))

n_norm = b + m * p_0 * T / T_0 # m*p_0*((3*A)/(R*T_0))+b

print('''
Berechneter Brechungsindex bei 1018hPa:
Luft: n = {} ± {}
Berechneter Brechungsindex bei Normalbedingungen:
Luft: n = {}
'''.format(np.mean(steig)*1018+np.mean(achse), np.std(achse)+np.std(steig), n_norm))

plt.errorbar(pressure, refindex1, 100*ref_index1_err, fmt='x', markersize=4, elinewidth=1, label=r'Messung 1 Fehler 100fach vergrößert.')
plt.plot(x, lin(*fit1_params, x), label='Ausgleichsgeraden')
plt.title(r'Druckverteilung des Brechungsindex von Luft')
plt.xlabel('Druck p/mbar')
plt.ylabel('Brechungsindex n')
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('Luft_1.pdf')
plt.close()
plt.errorbar(pressure, refindex2, 100*ref_index2_err, fmt='x', markersize=4, elinewidth=1, label=r'Messung 2 Fehler 100fach vergrößert.')
plt.plot(x, lin(*fit2_params, x), label='Ausgleichsgeraden')
plt.title(r'Druckverteilung des Brechungsindex von Luft')
plt.xlabel('Druck p/mbar')
plt.ylabel('Brechungsindex n')
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('Luft_2.pdf')
plt.close()
plt.errorbar(pressure, refindex3, 100*ref_index3_err, fmt='x', markersize=4, elinewidth=1, label=r'Messung 3 Fehler 100fach vergrößert.')
plt.plot(x, lin(*fit3_params, x), label='Ausgleichsgeraden')
plt.title(r'Druckverteilung des Brechungsindex von Luft')
plt.xlabel('Druck p/mbar')
plt.ylabel('Brechungsindex n')
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('Luft_3.pdf')
plt.close()
