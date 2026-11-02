# ligand_receptor.py

import numpy as np
import matplotlib.pyplot as plt

def Fact(G, S, Kd1, Kd2):
    """Fractional receptors activated by GLP1 and semaglutide"""
    return (Kd2 * G + Kd1 * S) / (Kd1 * Kd2 + Kd2 * G + Kd1 * S)

Kd1 = 0.17 # GLP-1 dissociation constant [nM]
Kd2 = 0.38 # Semaglutide dissociation constant [nM]
G_Fasting = 0.0150 # GLP-1 fasting concentration [nM]
G_Fed = 0.040 # GLP-1 fed concentration [nM]