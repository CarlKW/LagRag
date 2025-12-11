import numpy as np

N = 1000000  # number of simulated days

# Step 1: sample steel quality
steel_high = np.random.rand(N) < 0.25  # True = high-quality

# Step 2: choose lambda depending on steel quality
lam = np.where(steel_high, 10.0, 7.0)

# Step 3: sample production of clips and pins
clips = np.random.poisson(lam, size=N)
pins  = np.random.poisson(lam, size=N)

# Step 4: condition on the observation C=10, P=8
mask = (clips == 10) & (pins == 8)
posterior_estimate = steel_high[mask].mean()

print("Estimated P(S=High | C=10, P=8) =", posterior_estimate)