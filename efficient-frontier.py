import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Number of assets (20 in this case)
n_assets = 20

# Simulate random expected returns and covariance matrix for the 20 assets
np.random.seed(42)  # For reproducibility
mean_returns = np.array([
    68.96, 37.21, 35.19, 35.93, 45.93, 20.11, 22.15, 31.20, 28.80, 25.90, 21.97, 16.46, 39.31, 17.23, 23.57, 24.94, 25.53, 30.02, 20.54, 19.75
]) / 100
cov_matrix = np.array([
    [0.235162687, 0.084726298, 0.099739061, 0.140697766, 0.111595839, 0.040133614, 0.041496259, 0.080742254, 0.031116571, 0.035855986, 0.035117474, 0.033271256, 0.101677239, 0.049948578, 0.044969726, 0.041087001, 0.077677531, 0.018384744, 0.059405245, 0.061090843],
    [0.084726298, 0.158932221, 0.081700242, 0.074789287, 0.078079653, 0.03123852, 0.029464836, 0.051867885, 0.023980322, 0.035196402, 0.028754966, 0.029431274, 0.062564403, 0.036324632, 0.036792767, 0.035914437, 0.071343936, 0.018949326, 0.040672797, 0.045754595],
    [0.099739061, 0.081700242, 0.152102015, 0.091251822, 0.081929028, 0.03520577, 0.030990151, 0.069128076, 0.028064796, 0.040537761, 0.030346904, 0.027663627, 0.077074478, 0.038951191, 0.038268486, 0.038334082, 0.063343005, 0.01967048, 0.041524567, 0.04751522],
    [0.140697766, 0.074789287, 0.091251822, 0.184091951, 0.105144606, 0.038012363, 0.038856613, 0.066577452, 0.029160182, 0.040226761, 0.035419473, 0.034030352, 0.090548006, 0.049942378, 0.045633958, 0.037536087, 0.07586954, 0.020047885, 0.064869439, 0.058960092],
    [0.11159584, 0.07807965, 0.08192903, 0.10514461, 0.32367181, 0.0352906, 0.03397864, 0.07238973, 0.01618526, 0.04211466, 0.03006113, 0.03085724, 0.06957305, 0.04287765, 0.03253881, 0.03675137, 0.07672809, 0.01203466, 0.04843344, 0.04964709],
    [0.04013361, 0.03123852, 0.03520577, 0.03801236, 0.0352906, 0.08875866, 0.01807141, 0.02942097, 0.01377799, 0.02057367, 0.01717103, 0.01539413, 0.02852194, 0.01515597, 0.01719282, 0.01830545, 0.03106969, 0.01097614, 0.01581794, 0.02092641],
    [0.04149626, 0.02946484, 0.03099015, 0.03885661, 0.03397864, 0.01807141, 0.04635224, 0.03038871, 0.01911437, 0.02106381, 0.01994896, 0.01799042, 0.02883846, 0.02061107, 0.02237031, 0.01913869, 0.02530948, 0.01619568, 0.02224931, 0.02765489],
    [0.08074225, 0.05186789, 0.06912808, 0.06657745, 0.07238973, 0.02942097, 0.03038871, 0.10715435, 0.02043029, 0.02471338, 0.02265243, 0.01991396, 0.05554701, 0.03062375, 0.02844865, 0.02941652, 0.04663276, 0.01349795, 0.02838437, 0.0375923],
    [0.031116571, 0.023980322, 0.028064796, 0.029160182, 0.016185262, 0.013777991, 0.01911437, 0.020430285, 0.077059234, 0.025774288, 0.026947735, 0.02731079, 0.025652655, 0.017272537, 0.020256288, 0.019821713, 0.020260403, 0.020100473, 0.019706992, 0.016530799],
    [0.03585599, 0.0351964, 0.04053776, 0.04022676, 0.04211466, 0.02057367, 0.02106381, 0.02471338, 0.02577429, 0.14481821, 0.05520768, 0.05331289, 0.03254828, 0.02676196, 0.02756239, 0.02531357, 0.02733245, 0.02264988, 0.02769781, 0.03085139],
    [0.03511747, 0.02875497, 0.0303469, 0.03541947, 0.03006113, 0.01717103, 0.01994896, 0.02265243, 0.02694774, 0.05520768, 0.06692036, 0.05171929, 0.02665003, 0.02411265, 0.0258721, 0.02486433, 0.02478695, 0.02367203, 0.02713862, 0.02800607],
    [0.03327126, 0.02943127, 0.02766363, 0.03403035, 0.03085724, 0.01539413, 0.01799042, 0.01991396, 0.02731079, 0.05331289, 0.05171929, 0.08933165, 0.02784982, 0.02426574, 0.02490761, 0.02533264, 0.02366736, 0.02546933, 0.0309967, 0.02732794],
    [0.101677239, 0.062564403, 0.077074478, 0.090548006, 0.069573052, 0.028521941, 0.028838457, 0.055547007, 0.025652655, 0.032548277, 0.026650027, 0.027849821, 0.179679924, 0.037911375, 0.043195801, 0.032241518, 0.052015046, 0.019650855, 0.047553309, 0.045984249],
    [0.049948578, 0.036324632, 0.038951191, 0.049942378, 0.042877645, 0.015155973, 0.020611066, 0.030623746, 0.017272537, 0.026761963, 0.02411265, 0.024265741, 0.037911375, 0.077289923, 0.028599301, 0.021804835, 0.038829644, 0.019278224, 0.035864443, 0.037086244],
    [0.044969726, 0.036792767, 0.038268486, 0.045633958, 0.032538813, 0.017192822, 0.022370315, 0.028448646, 0.020256288, 0.027562385, 0.025872097, 0.024907612, 0.043195801, 0.028599301, 0.061645412, 0.024728284, 0.036329868, 0.022744528, 0.033113453, 0.031118298],
    [0.041087001, 0.035914437, 0.038334082, 0.037536087, 0.036751366, 0.018305446, 0.019138687, 0.029416517, 0.019821713, 0.025313572, 0.024864331, 0.025332641, 0.032241518, 0.021804835, 0.024728284, 0.068854558, 0.031045659, 0.017879357, 0.024668953, 0.025481663],
    [0.077677531, 0.071343936, 0.063343005, 0.07586954, 0.07672809, 0.03106969, 0.025309478, 0.046632758, 0.020260403, 0.027332448, 0.024786953, 0.023667363, 0.052015046, 0.038829644, 0.036329868, 0.031045659, 0.231793972, 0.01983331, 0.045675698, 0.043570385],
    [0.018384744, 0.018949326, 0.01967048, 0.020047885, 0.012034656, 0.010976137, 0.016195676, 0.013497952, 0.020100473, 0.022649875, 0.023672028, 0.025469326, 0.019650855, 0.019278224, 0.022744528, 0.017879357, 0.01983331, 0.058660102, 0.024110319, 0.021817954],
    [0.059405245, 0.040672797, 0.041524567, 0.064869439, 0.048433442, 0.015817943, 0.022249315, 0.02838437, 0.019706992, 0.027697807, 0.02713862, 0.030996698, 0.047553309, 0.035864443, 0.033113453, 0.024668953, 0.045675698, 0.024110319, 0.112628651, 0.045038935],
    [0.061090843, 0.045754595, 0.04751522, 0.058960092, 0.049647087, 0.020926409, 0.02765489, 0.037592304, 0.016530799, 0.030851389, 0.028006071, 0.027327944, 0.045984249, 0.037086244, 0.031118298, 0.025481663, 0.043570385, 0.021817954, 0.045038935, 0.094700985]
])

#cov_matrix = np.dot(cov_matrix, cov_matrix.T)  # Create a positive semi-definite covariance matrix

# Function to calculate portfolio performance (return and volatility) based on weights
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns) / 100
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

# Function to generate random portfolios
def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix):
    results = np.zeros((3, num_portfolios))  # Store returns, volatility, and Sharpe ratio
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)  # Normalize weights to sum to 1
        
        returns, volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe_ratio = returns / volatility
        
        results[0, i] = returns
        results[1, i] = volatility
        results[2, i] = sharpe_ratio
        
        weights_record.append(weights)
    
    return results, weights_record

# Function to minimize volatility for a given return target
from scipy.optimize import minimize

def minimize_volatility(mean_returns, cov_matrix, target_return):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    # Initial weights
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights must sum to 1
                   {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target_return})  # Target return constraint
    bounds = tuple((-2, 2) for _ in range(num_assets))  # Weights must be between 0 and 1
    result = minimize(lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1],  
                      num_assets * [1. / num_assets], 
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints)
    return result.x

# Given weights
base_weights = np.array([
    0.255136427, 0.021760932, 0, 0, 0.039913067, 0.015645052, 0.009446249,
    0, 0.18102215, 0.013964171, 0, 0, 0.003792317, 0, 0, 0.054672413,
    0, 0.404647222, 0, 0
])

# Function to generate nearby weights
def generate_perturbed_weights(base_weights, num_points=100, perturbation=0.02):
    """
    Generate weights by perturbing the given base weights.
    :param base_weights: Array of base weights
    :param num_points: Number of perturbed weights to generate
    :param perturbation: Maximum perturbation applied to each weight
    :return: List of perturbed weights
    """
    perturbed_weights = []
    for _ in range(num_points):
        noise = np.random.uniform(-perturbation, perturbation, size=base_weights.shape)
        new_weights = base_weights + noise
        new_weights = np.clip(new_weights, 0, None)  # Ensure weights are non-negative
        new_weights /= np.sum(new_weights)  # Normalize weights to sum to 1
        perturbed_weights.append(new_weights)
    return np.array(perturbed_weights)

def markowitz_weights(mean_returns, cov_matrix, target_return=None, risk_free_rate=0.02415, maximize_sharpe=False):
    """
    Compute the Markowitz portfolio weights.
    
    Parameters:
    - mean_returns: Array of expected returns for the assets.
    - cov_matrix: Covariance matrix of asset returns.
    - target_return: Target portfolio return. If None and maximize_sharpe is False, 
      the weights minimize volatility without targeting a specific return.
    - risk_free_rate: Risk-free rate for Sharpe ratio maximization.
    - maximize_sharpe: Boolean. If True, computes the weights that maximize the Sharpe ratio.
    
    Returns:
    - weights: Optimal weights for the portfolio.
    """
    num_assets = len(mean_returns)

    # Define the objective function
    if maximize_sharpe:
        # Objective: Maximize Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - risk_free_rate) / portfolio_volatility
    else:
        # Objective: Minimize portfolio volatility
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights must sum to 1

    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return})  # Target return

    # Bounds: weights between 0 and 1 (modify to (-1, 1) for short selling)
    bounds = tuple((-1, 1) for _ in range(num_assets))

    # Initial guess: equal allocation
    initial_weights = num_assets * [1. / num_assets]

    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    optimal_weights = result.x
    portfolio_return = np.dot(optimal_weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    max_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

    return optimal_weights, max_sharpe


# Generate perturbed weights
num_points = 10000
perturbation = 0.75
perturbed_weights = generate_perturbed_weights(base_weights, num_points, perturbation)

def compute_efficient_frontier(mean_returns, cov_matrix, num_points=100):
    """
    Compute the efficient frontier by varying target returns.
    """
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
    frontier_volatility = []
    frontier_weights = []
    
    for target in target_returns:
        # Minimize volatility for each target return
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}
        )

        bounds = tuple((0, 1) for _ in range(len(mean_returns)))
        
        result = minimize(portfolio_volatility, len(mean_returns) * [1. / len(mean_returns)],
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            frontier_volatility.append(result.fun)
            frontier_weights.append(result.x)
    
    return target_returns, frontier_volatility, frontier_weights

def compute_efficient_frontier_shorting(mean_returns, cov_matrix, num_points=100):
    """
    Compute the efficient frontier by varying target returns.
    """
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
    frontier_volatility = []
    frontier_weights = []
    
    for target in target_returns:
        # Minimize volatility for each target return
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}
        )
        bounds = tuple((-1, 1) for _ in range(len(mean_returns)))
        
        result = minimize(portfolio_volatility, len(mean_returns) * [1. / len(mean_returns)],
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            frontier_volatility.append(result.fun)
            frontier_weights.append(result.x)
    
    return target_returns, frontier_volatility, frontier_weights



# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

# Calculate returns and volatilities for the perturbed portfolios
perturbed_returns = []
perturbed_volatilities = []

for weights in perturbed_weights:
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    perturbed_returns.append(ret)
    perturbed_volatilities.append(vol)

perturbed_sharpe_ratios = [
    (ret - 0.02415) / vol if vol > 0 else 0  # Avoid division by zero
    for ret, vol in zip(perturbed_returns, perturbed_volatilities)
]

# Generate random portfolios
#num_portfolios = 5000000
#results, weights = generate_random_portfolios(num_portfolios, mean_returns, cov_matrix)

# Compute the efficient frontier

risk_free_rate = 0.02415  # Replace with your risk-free rate (e.g., 1%)
weights, max_sharpe = markowitz_weights(mean_returns, cov_matrix, risk_free_rate=risk_free_rate, maximize_sharpe=True)
print("Maximum Sharpe Ratio Portfolio Weights:", weights)
print("Maximum Sharpe Ratio:", max_sharpe)

target_returns_no_short, frontier_volatility_no_short, _ = compute_efficient_frontier(mean_returns, cov_matrix)
target_returns_short, frontier_volatility_short, _ = compute_efficient_frontier_shorting(mean_returns, cov_matrix)

def cal_line(risk_free_rate, tangency_return, tangency_volatility, x_range):
    sharpe_ratio = (tangency_return - risk_free_rate) / tangency_volatility
    return risk_free_rate + sharpe_ratio * x_range

cal_volatility_range = np.linspace(0, max(perturbed_volatilities), 500)
tangency_return = 0.399978354
tangency_volatility = 0.217190724
cal_return = cal_line(risk_free_rate, tangency_return, tangency_volatility, cal_volatility_range)

# Plot the efficient frontier and random portfolios
plt.figure(figsize=(10, 6))
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.plot(frontier_volatility_no_short, target_returns_no_short, 'r--', label='Efficient Frontier No Shorting')
plt.plot(frontier_volatility_short, target_returns_short, 'r--', color='blue', label='Efficient Frontier With Shorting')
plt.scatter(perturbed_volatilities, perturbed_returns, color='blue', label='Random Portfolios', zorder=3)
plt.scatter(0.217190724, 0.399978354, color='#E63946', label='Tangency Portfolio No Short', zorder=5)
plt.scatter(0.244532168, 0.479565293, color='#2A9D8F', label='Markowitz Portfolio With Short', zorder=5)
plt.scatter(0.209712265, 0.369774633, color='#F4A261', label='Our Portfolio With Min 2% and Max 25%', zorder=5)
plt.scatter(0.205002937, 0.295351119, color='magenta', label='Equally Weighted', zorder=5)
plt.scatter(0.294556569, 0.450680994, color='yellow', label='Market Cap Weighted', zorder=5)
plt.plot(
    cal_volatility_range, 
    cal_return, 
    color='green', 
    linestyle='--', 
    linewidth=2, 
    label='Capital Allocation Line (CAL)'
)
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Return')
plt.title('Return and Volatility of all portfolios')
plt.legend(loc='upper right')
plt.show()