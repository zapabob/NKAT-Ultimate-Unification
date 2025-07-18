import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpmath import zetazero

zeros = [float(zetazero(n).imag) for n in tqdm(range(1, 101))]
plt.figure(figsize=(8,4))
plt.plot(range(1, 101), zeros, marker='o')
plt.xlabel('Zero Index')
plt.ylabel('Im part of Zeta Zero')
plt.title('Riemann Zeta Zeros (First 100)')
plt.grid(True)
plt.tight_layout()
plt.savefig('../../figures/riemann_zeros.png')
plt.show()
# Caption: Visualization of the imaginary parts of the first 100 nontrivial zeros of the Riemann zeta function. 