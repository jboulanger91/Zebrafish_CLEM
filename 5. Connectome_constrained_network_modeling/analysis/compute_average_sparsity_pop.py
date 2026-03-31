import numpy as np

# Compute the average sparsity across and within populations
# This average constraint is applied to the free popluation of helper neurons

# population-specific sparsity constraints
to_pop_constraints = {
    "A": {"N": 15,
          "E_frac": 0.7,
          "sparsity_from_A": [0.1125, 0],  # [L, R]
          "sparsity_from_B": [0, 0.2],
          "sparsity_from_C": [0.08, 0.03],
          "sparsity_from_D": [0.02, 0.05]},
    "B": {"N": 15,
          "E_frac": 0.1,
          "sparsity_from_A": [0.18, 0],
          "sparsity_from_B": [0, 0.3],
          "sparsity_from_C": [0.45, 0.15],
          "sparsity_from_D": [0, 0.05]},
    "C": {"N": 2,
          "E_frac": 0.1,
          "sparsity_from_A": [0.475, 0],
          "sparsity_from_B": [0, 0.04],
          "sparsity_from_C": [0.09, 0],
          "sparsity_from_D": [0, 0.05]},
    "D": {"N": 11,
          "E_frac": 1/3,
          "sparsity_from_A": [0.065, 0],
          "sparsity_from_B": [0, 0.06],
          "sparsity_from_C": [0.04, 0],
          "sparsity_from_D": [0, 0]},
}
N_tot = np.sum([p["N"] for p in to_pop_constraints.values()])

print("AVERAGE CONSTRAINTS\n")
average_constraints = {}
average_constraints["E_frac"] = np.sum([p["E_frac"] * p["N"] for p in to_pop_constraints.values()]) / N_tot
print(f"E_frac | {average_constraints['E_frac']}\n")
for pop in to_pop_constraints.keys():
    average_constraints[f"sparsity_from_{pop}"] = np.sum([np.mean(p[f"sparsity_from_{pop}"]) * p["N"] for p in to_pop_constraints.values()]) / N_tot
    average_constraints[f"sparsity_to_{pop}"] = 0
    for k, v in to_pop_constraints[pop].items():
        if "sparsity_from" in k:
            pop_from = k[-1]
            average_constraints[f"sparsity_to_{pop}"] += np.sum([np.mean(v) * to_pop_constraints[pop_from]["N"] ])
    average_constraints[f"sparsity_to_{pop}"] /= N_tot
    print(f"{f'Sparsity from {pop}'} | {average_constraints[f'sparsity_from_{pop}']}")
    print(f"{f'Sparsity to {pop}'} | {average_constraints[f'sparsity_to_{pop}']}\n")


average_constraints["sparsity_within"] = np.sum([np.mean(p[f"sparsity_from_{k}"]) * p["N"] for k, p in to_pop_constraints.items()]) / N_tot
print(f"{f'Sparsity within'} | {average_constraints['sparsity_within']}\n")