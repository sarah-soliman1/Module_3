
import numpy as np
import matplotlib.pyplot as plt
from ligand_receptor import Fact, Kd1, Kd2, G_Fasting, G_Fed

S = np.logspace(-3, 2, 500) # Semaglutide concentration [nM]

plt.figure()

plt.semilogx(S, Fact(G_Fasting, S, Kd1, Kd2),
            label = "Fasting GLP-1")
plt.semilogx(S, Fact(G_Fed, S, Kd1, Kd2),
            label = "Fed GLP-1")

plt.xlabel("Semaglutide concentration [nM]")
plt.ylabel("Fraction of activated receptors")
plt.title("GLP_1 and Semaglutide GLP1R Activation")

plt.xlim(0.001, 100)
plt.ylim(0,1)
plt.legend()
plt.grid()

plt.show()