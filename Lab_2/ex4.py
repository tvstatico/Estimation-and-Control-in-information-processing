import matplotlib.pyplot as plt

# --- Configuration ---
# 17 values measured across 17 years
r_sequence = [2, 2.5, 1, 1.2, 3.1, 0.5, 4, 4.4, 3, 2.9, 2.8, 1.9, 1.5, 1.4, 7, 3.8, 8]
history = []
x = 0.5  # Initial population at Year 0

# --- Simulation ---
print("Yearly Population Log:")
for year, r in enumerate(r_sequence):
    history.append(x)
    print(f"Year {year}: x = {x:.4f} (r for next year = {r})")
    
    # Calculate next year
    x = r * x * (1 - x)

history.append(x)
print(f"Year 17: x = {x:.4f}")

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.plot(range(18), history, marker='o', color='tab:red')
plt.title("Ex4: Population over 17 Years with Changing r")
plt.xlabel("Year")
plt.ylabel("Population Size")
plt.grid(True)
plt.show()

# In the 15th year (index 14), r = 7. The population size gets pushed into negative 
# numbers. By the 17th year, the mathematical model diverges entirely to a massive 
# negative number. In a real-world context, the population went entirely 
# extinct the moment x dropped below 0. 
