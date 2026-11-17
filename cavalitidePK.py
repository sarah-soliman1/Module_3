import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===== choose product: 'Ozempic' or 'Wegovy' =====
drug = 'Ozempic'
#drug = 'Wegovy'


# product-specific parameters

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

p = products[drug]
Favailable, ka, D = p['Favailable'], p['ka'], p['D']
interval, n_doses, depot = p['interval'], p['n_doses'], p['depot']

# shared model parameters
ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # rest of semaglutide bound to albumin

# simulation parameters
dt = 0.1        # h, max intigration step size for solve_ivp

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

# --- run ---
# Ozempic
p = products['Ozempic']
Favailable = p["Favailable"]
ka = p["ka"]

t_oz, Asc_oz, Cc_oz = simulate(
    p["D"], p['n_doses'], p['interval'], dt
)

# Wegovy
p = products['Wegovy']
Favailable = p['Favailable']
ka = p['ka']

t_we, Asc_we, Cc_we = simulate(
    p['D'], p['n_doses'], p['interval'], dt
)

plt.figure(figsize = (10,6))
plt.plot(t_oz / 24, Ffree*Cc_oz, label="Ozempic Weekly Injection")
plt.plot(t_we / 24, Ffree*Cc_we, label="Wegovy Daily Pill")
plt.xlabel("Time (days)")
plt.ylabel("Free blood concentration (nM)")
plt.title('Ozempic vs Wegovy')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

p = products['Cavalitide']
Favailable = p['Favailable']
ka = p['ka']

t_cav, Asc_cav, Cc_cav = simulate(
    p['D'], p['n_doses'], p['interval'], dt
)

plt.figure(figsize = (10,6))
plt.plot(t / 24, Ffree*Cc_cav, label="Cavalitide 25mg Daily")
plt.xlabel("Time (days)")
plt.ylabel("Blood concentration [nM]")
plt.title("Cavalitide: 25mg Daily")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

optimized_D = 20.8
t_opt, Asc_opt, Cc_opt = simulate(
    optimized_D, p['n_doses'], p['interval'], dt
)

last_interval = t_opt >= (t_opt[-1] - p['interval'])
steady_state_avg = np.trapezoid(
    Ffree * Cc_opt[last_interval],
    t_opt[last_interval]
) / p['interval']

print("Optimized daily dose:", optimized_D, "mg")
print("Average steady-state free concentration:",
      round(steady_state_avg, 3), "nM")

plt.figure(figsize = (10,6))
plt.plot(t_opt / 24, Ffree*Cc_opt, label="Cavalitide 20.8mg Daily")
plt.xlabel("Time (days)")
plt.ylabel("Blood Concentration (nM)")
plt.title("Optimized Cavalitide Dose")
plt.legend()
plt.grid(alpha=0.3)
plt.show()