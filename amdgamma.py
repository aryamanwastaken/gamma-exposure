import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load the AMD data
filename = 'amd_quotedata.csv'
data = pd.read_csv(filename)

# Define the Black-Scholes gamma calculation function
def black_scholes_gamma(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# Calculate gamma for each option in the dataset
data['Gamma'] = data.apply(lambda row: black_scholes_gamma(
    S=row['SpotPrice'],
    K=row['StrikePrice'],
    T=row['TimeToExpiration'],
    r=row['RiskFreeRate'],
    sigma=row['Volatility'],
    option_type=row['OptionType']
), axis=1)

# Calculate net gamma exposure for each option
data['NetGammaExposure'] = data['Gamma'] * data['OpenInterest']

# Plot the net gamma exposure profile
plt.plot(data['StrikePrice'], data['NetGammaExposure'])
plt.title('Net Gamma Exposure Profile for AMD')
plt.xlabel('Strike Price')
plt.ylabel('Net Gamma Exposure')
plt.grid(True)
plt.show()
