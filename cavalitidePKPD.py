#Step 1: Import python extensions
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

#Step 2: Define drug variable and parameters
drug = 'Ozempic'
#drug = 'Wegovy'

products = {
    'Ozempic': dict(
        Favailable=0.89,
        ka=0.029,
        D=2.0,
        interval=168,
        n_doses=8,
        depot='subcutaneous'
    ),


    'Wegovy': dict(
        Favailable=0.01,
        ka=0.0029,
        D=25,
        interval=24,
        n_doses=100,
        depot='gut (oral)'
    ),


    'Cavalitide': dict(
        Favailable=0.01 ,
        ka=0.006,
        D=25,
        interval=24,
        n_doses=100,
        depot='oral'
    )
}

def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists L1 and L2"""
    return (Kd2*L1 + Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)


p = products[drug]
Favailable, ka, D = p['Favailable'], p['ka'], p['D']
interval, n_doses, depot = p['interval'], p['n_doses'], p['depot']

ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # rest of semaglutide bound to albumin
#in [nM]
Kd1 = 0.17      # L1, GLP1
Kd2_sema = 0.38 # semaglutide (reference)
L1 = 0.015      # GLP1 (fasting)
L2 = np.logspace(-3, 2, 500)

# simulation parameters
dt = 0.1        # h, Euler step size


#Step 3: Defining ODE and simulation functions
def ODEfunc(t, y):
    Asc, Cc = y
    dAsc = -ka*Asc            # Asc: depot amount (mg)
    dCc = ka*Asc/V - ke*Cc    # Cc: blood concentration (mg/L)
    return [dAsc, dCc]


def simulate(D, n_doses, interval, dt):
    """Repeated dosing every `interval` h; returns time (h), depot amount (mg),
    and blood concentration (nmol/L)."""
    t_out, Asc_out, Cc_out = [], [], []
    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt))      # Euler steps per dosing interval
    for i in range(n_doses):
        y[0] += Favailable*D                 # absorbed fraction of dose enters the depot
        t0 = i * interval
        t_seg = t0 + dt * np.arange(n_steps + 1)
        y_seg = np.zeros((2, n_steps + 1))
        y_seg[:, 0] = y
       
        # explicit Euler: y[n+1] = y[n] + dt * f(t[n], y[n])
        solution = solve_ivp(
            ODEfunc,
            (t0, t0 + interval),
            y,
            t_eval = t_seg,
            max_step = dt
        )


        y_seg = solution.y
           
        t_out.append(t_seg)
        Asc_out.append(y_seg[0])
        Cc_out.append(y_seg[1])
        y = y_seg[:, -1].copy()         # carry state into next interval
    t = np.concatenate(t_out)
    Asc_out = np.concatenate(Asc_out)
    Cc_out = np.concatenate(Cc_out)
    Cc_nM = Cc_out * 1e-3 / MW * 1e9
    return t, Asc_out, Cc_nM

# Step 3: Load in cavalitide data from csv file
data = np.loadtxt("C:/Users/sarah/BME 2315/Module 3/Module_3/cavalitide_data.csv", delimiter=",", skiprows=1)
L2_loaded, Fact_loaded = data[:, 0], data[:, 1]


# Step 4: Brute force method
Kd2_candidates = np.logspace(-3, 3, 2000)
SSEs = [np.sum((Fact_loaded - Fact(L1, L2_loaded, Kd1, k))**2) for k in Kd2_candidates]
i = int(np.argmin(SSEs))
Kd2_brute, SSE_brute = Kd2_candidates[i], SSEs[i]
print(f"Brute-force Kd2 = {Kd2_brute:.4f} nM (SSE = {SSE_brute:.5f})")


# Step 5: Curve_fit method
popt, pcov = curve_fit(lambda L2, Kd2: Fact(L1, L2, Kd1, Kd2),
                       L2_loaded, Fact_loaded, p0=[Kd2_brute], bounds=(0, np.inf))
Kd2_fit = popt[0]
SSE_fit = np.sum((Fact_loaded - Fact(L1, L2_loaded, Kd1, Kd2_fit))**2)
print(f"Fitted Kd2 = {Kd2_fit:.4f} +/- {np.sqrt(pcov[0,0]):.4f} nM (SSE = {SSE_fit:.5f})")


# Step 6: Run stimulation
t, Asc, Cc = simulate(D, n_doses, interval, dt)


# Step 7: Define and print plots
plt.plot(t / 24, Asc, label=f"{depot} depot [mg]")
plt.plot(t / 24, Ffree*Cc, label="free in blood [nM]")
plt.xlabel("Time (days)")
plt.ylabel("Cavalitide")
plt.title(f"{drug}: {D} mg every {interval} h, {n_doses} doses")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize = (10,6))
plt.plot(t / 24, Ffree*Cc, label="Cavalitide free in blood [nM]")
plt.xlabel("Time (days)")
plt.ylabel("Blood concentration [nM]")
plt.title("Cavalitide: 25mg Daily")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize = (10,6))
plt.plot(t / 24, Ffree*Cc, label="Cavalitide 20.8mg Daily")
plt.xlabel("Time (days)")
plt.ylabel("Blood Concentration (nM)")
plt.title("Optimized Cavalitide Dose")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(7, 4.5))
plt.semilogx(L2_loaded, Fact_loaded, "ko", label="experimental data")
plt.semilogx(L2, Fact(L1, L2, Kd1, Kd2_brute), ":", lw=2.5,
             label=f"brute force (Kd = {Kd2_brute:.3f} nM)")
plt.semilogx(L2, Fact(L1, L2, Kd1, Kd2_fit), "--", lw=2,
             label=f"curve_fit (Kd = {Kd2_fit:.3f} nM)")
plt.semilogx(L2, Fact(L1, L2, Kd1, Kd2_sema), "-", lw=1.2, alpha=0.5,
             label=f"cavalitide ref. (Kd = {Kd2_sema} nM)")
plt.ylim(-0.1, 1.1)
plt.xlabel("[Cavalitide] (nM)")
plt.ylabel("Fraction active receptors (Fact)")
plt.title("GLP1R activation by Cavalitide with GLP1 present")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cavalitide_fit.png", dpi=150)
plt.show()
