import numpy as np
from skfuzzy import gaussmf

from utils.utils import plot_func, plot_funcs, gen_mf_by_range, plotMFs

# varss = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
#
#
# mf_gauss = [
#     [{"mean": 4.3, "sigma": 0.6},
#      {"mean": 6.1, "sigma": 0.6},
#      {"mean": 7.9, "sigma": 0.6}],
#
#     [{"mean": 2.0, "sigma": 0.4},
#      {"mean": 3.2, "sigma": 0.4},
#      {"mean": 4.4, "sigma": 0.4}],
#
#     [{"mean": 1.0, "sigma": 1},
#      {"mean": 3.95, "sigma": 1},
#      {"mean": 6.9, "sigma": 1}],
#
#     [{"mean": 0.1, "sigma": 0.4},
#      {"mean": 1.3, "sigma": 0.4},
#      {"mean": 2.5, "sigma": 0.4}]
# ]
#
# plotMFs(gaussmf, varss, mf_gauss)

gen_mf_by_range("gaussian", 3, 4.3, 7.9)
gen_mf_by_range("gaussian", 3, 2.0, 4.4)
gen_mf_by_range("gaussian", 3, 1.0, 6.9)
gen_mf_by_range("gaussian", 3, 0.1, 2.5)
