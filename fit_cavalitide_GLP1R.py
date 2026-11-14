#fit_cavalitide_GLP1R.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists L1 and L2"""
    return (Kd2*L1 + Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)

# Parameters [nM]
Kd1 = 0.17      # L1, GLP1
Kd2_sema = 0.38 # semaglutide (reference)
L1 = 0.015      # GLP1 (fasting)
L2 = np.logspace(-3, 2, 500)

# Load the data (edit path as needed)
data = np.loadtxt("cavalitide_data.csv", delimiter=",", skiprows=1)
L2_loaded, Fact_loaded = data[:, 0], data[:, 1]

# Approach 1: brute force
Kd2_candidates = np.logspace(-3, 3, 2000)
SSEs = [np.sum((Fact_loaded - Fact(L1, L2_loaded, Kd1, k))**2) for k in Kd2_candidates]
i = int(np.argmin(SSEs))
Kd2_brute, SSE_brute = Kd2_candidates[i], SSEs[i]
print(f"Brute-force Kd2 = {Kd2_brute:.4f} nM (SSE = {SSE_brute:.5f})")

# Approach 2: curve_fit
popt, pcov = curve_fit(lambda L2, Kd2: Fact(L1, L2, Kd1, Kd2),
                       L2_loaded, Fact_loaded, p0=[Kd2_brute], bounds=(0, np.inf))
Kd2_fit = popt[0]
SSE_fit = np.sum((Fact_loaded - Fact(L1, L2_loaded, Kd1, Kd2_fit))**2)
print(f"Fitted Kd2 = {Kd2_fit:.4f} +/- {np.sqrt(pcov[0,0]):.4f} nM (SSE = {SSE_fit:.5f})")

plt.figure(figsize=(7, 4.5))
plt.semilogx(L2_loaded, Fact_loaded, "ko", label="experimental data")
plt.semilogx(L2, Fact(L1, L2, Kd1, Kd2_brute), ":", lw=2.5,
             label=f"brute force (Kd = {Kd2_brute:.3f} nM)")
plt.semilogx(L2, Fact(L1, L2, Kd1, Kd2_fit), "--", lw=2,
             label=f"curve_fit (Kd = {Kd2_fit:.3f} nM)")
plt.semilogx(L2, Fact(L1, L2, Kd1, Kd2_sema), "-", lw=1.2, alpha=0.5,
             label=f"semaglutide ref. (Kd = {Kd2_sema} nM)")
plt.ylim(-0.1, 1.1)
plt.xlabel("[Cavalitide] (nM)")
plt.ylabel("Fraction active receptors (Fact)")
plt.title("GLP1R activation by Cavalitide with GLP1 present")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cavalitide_fit.png", dpi=150)
plt.show()