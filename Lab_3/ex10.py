import numpy as np

np.random.seed(42)
N = 100000

# --- Simulation ---
# Simulate two independent events: A (40% chance) and B (50% chance)
prob_A = 0.40
prob_B = 0.50

event_A = np.random.rand(N) < prob_A
event_B = np.random.rand(N) < prob_B

# --- Computation ---
# P(A | B) = P(A and B) / P(B)
A_and_B = np.sum(event_A & event_B)
total_B = np.sum(event_B)

P_A_given_B = A_and_B / total_B
P_A_empirical = np.sum(event_A) / N

print(f"Empirical P(A): {P_A_empirical:.4f}")
print(f"Estimated P(A|B): {P_A_given_B:.4f}")

# The calculated conditional probability P(A|B) is functionally identical to the 
# marginal probability P(A). This statistical parity demonstrates that events A 
# and B are independent; knowing that event B has occurred provides no predictive 
# information regarding the likelihood of event A.