# Chapter 1
# Portfolio returns during the crisis
# Select portfolio asset prices for the middle of the crisis, 2008-2009
asset_prices = portfolio.loc['2008-01-01':'2009-12-31']

# Plot portfolio's asset prices during this time
asset_prices.plot().set_ylabel("Closing Prices, USD")
plt.show()

#2
# Compute the portfolio's daily returns
asset_returns = asset_prices.pct_change()
portfolio_returns = asset_returns.dot(weights)

# Plot portfolio returns
portfolio_returns.plot().set_ylabel("Daily Return, %")
plt.show()

# Asset covariance and portfolio volatility
# Generate the covariance matrix from portfolio asset's returns
covariance = asset_returns.cov()

# Annualize the covariance using 252 trading days per year
covariance = covariance * 252

# Display the covariance matrix
print(covariance)

#2
# Compute and display portfolio volatility for 2008 - 2009
portfolio_variance = np.transpose(weights) @ covariance @ weights
portfolio_volatility = np.sqrt(portfolio_variance)
print(portfolio_volatility)

#3
# Calculate the 30-day rolling window of portfolio returns
returns_windowed = portfolio_returns.rolling(30)

# Compute the annualized volatility series
volatility_series = returns_windowed.std()*np.sqrt(252)

# Plot the portfolio volatility
volatility_series.plot().set_ylabel("Annualized Volatility, 30-day Window")
plt.show()

# Risk factors and the financial crisis
# Frequency resampling primer
# Convert daily returns to quarterly average returns
returns_q = returns.resample('Q').mean()

# Examine the beginning of the quarterly series
print(returns_q.head())

# Now convert daily returns to weekly minimum returns
returns_w = returns.resample('W').min()

# Examine the beginning of the weekly series
print(returns_w.head())

# Visualizing risk factor correlation
# Transform the daily portfolio_returns into quarterly average returns
portfolio_q_average = portfolio_returns.resample('Q').mean().dropna()

# Create a scatterplot between delinquency and quarterly average returns
plot_average.scatter(mort_del, portfolio_q_average)

# Transform daily portfolio_returns returns into quarterly minimum returns
portfolio_q_min = portfolio_returns.resample('Q').min().dropna()

# Create a scatterplot between delinquency and quarterly minimum returns
plot_min.scatter(mort_del, portfolio_q_min)
plt.show()

# Least-squares factor model
# Add a constant to the regression
mort_del = sm.add_constant(mort_del)

# Create the regression factor model and fit it to the data
results = sm.OLS(port_q_mean, mort_del).fit()

# Print a summary of the results
print(results.summary())

#2
# Add a constant to the regression
mort_del = sm.add_constant(mort_del)

# Create the regression factor model and fit it to the data
results = sm.OLS(port_q_min, mort_del).fit()

# Print a summary of the results
print(results.summary())

#3
# Add a constant to the regression
mort_del = sm.add_constant(mort_del)

# Create the regression factor model and fit it to the data
results = sm.OLS(vol_q_mean, mort_del).fit()

# Print a summary of the results
print(results.summary())

# Modern portfolio theory
# Practice with PyPortfolioOpt: returns
# Load the investment portfolio price data into the price variable.
prices = pd.read_csv("portfolio.csv")

# Convert the 'Date' column to a datetime index
prices['Date'] = pd.to_datetime(prices['Date'], format='%d/%m/%Y')
prices.set_index(['Date'], inplace = True)

#2
# Import the mean_historical_return method
from pypfopt.expected_returns import mean_historical_return

# Compute the annualized average historical return
mean_returns = mean_historical_return(prices, frequency = 252)

# Plot the annualized average historical return
plt.plot(mean_returns, linestyle = 'None', marker = 'o')
plt.show()

# Practice with PyPortfolioOpt: covariance
# Import the CovarianceShrinkage object
from pypfopt.risk_models import CovarianceShrinkage

# Create the CovarianceShrinkage instance variable
cs = CovarianceShrinkage(prices)

#2
# Compute the sample covariance matrix of returns
sample_cov = prices.pct_change().cov() * 252

# Compute the efficient covariance matrix of returns
e_cov = cs.ledoit_wolf()

# Display both the sample covariance_matrix and the efficient e_cov estimate
print("Sample Covariance Matrix\n", sample_cov, "\n")
print("Efficient Covariance Matrix\n", e_cov, "\n")

# Breaking down the financial crisis
# Create a dictionary of time periods (or 'epochs')
epochs = { 'before' : {'start': '1-1-2005', 'end': '31-12-2006'},
           'during' : {'start': '1-1-2007', 'end': '31-12-2008'},
           'after'  : {'start': '1-1-2009', 'end': '31-12-2010'}
         }

# Compute the efficient covariance for each epoch
e_cov = {}
for x in epochs.keys():
  sub_price = prices.loc[epochs[x]['start']:epochs[x]['end']]
  e_cov[x] = CovarianceShrinkage(sub_price).ledoit_wolf()

# Display the efficient covariance matrices for all epochs
print("Efficient Covariance Matrices\n", e_cov)

# The efficient frontier and the financial crisis
# Initialize the Crtical Line Algorithm object
efficient_portfolio_during = CLA(returns_during, ecov_during)

# Find the minimum volatility portfolio weights and display them
print(efficient_portfolio_during.min_volatility())

# Compute the efficient frontier
(ret, vol, weights) = efficient_portfolio_during.efficient_frontier()

# Add the frontier to the plot showing the 'before' and 'after' frontiers
plt.scatter(vol, ret, s = 4, c = 'g', marker = '.', label = 'During')
plt.legend()
plt.show()

# Chapter 2
# Measuring Risk
# VaR for the Normal distribution
# Create the VaR measure at the 95% confidence level using norm.ppf()
VaR_95 = norm.ppf(0.95)

# Create the VaR meaasure at the 5% significance level using numpy.quantile()
draws = norm.rvs(size = 100000)
VaR_99 = np.quantile(draws, 0.99)

# Compare the 95% and 99% VaR
print("95% VaR: ", VaR_95, "; 99% VaR: ", VaR_99)

# Plot the normal distribution histogram and 95% VaR
plt.hist(draws, bins = 100)
plt.axvline(x = VaR_95, c='r', label = "VaR at 95% Confidence Level")
plt.legend(); plt.show()

# Comparing CVaR and VaR
# Compute the mean and variance of the portfolio losses
pm = portfolio_losses.mean()
ps = portfolio_losses.std()

# Compute the 95% VaR using the .ppf()
VaR_95 = norm.ppf(0.95, loc = pm, scale = ps)
# Compute the expected tail loss and the CVaR in the worst 5% of cases
tail_loss = norm.expect(lambda x: x, loc = pm, scale = ps, lb = VaR_95)
CVaR_95 = (1 / (1 - 0.95)) * tail_loss

# Plot the normal distribution histogram and add lines for the VaR and CVaR
plt.hist(norm.rvs(size = 100000, loc = pm, scale = ps), bins = 100)
plt.axvline(x = VaR_95, c='r', label = "VaR, 95% confidence level")
plt.axvline(x = CVaR_95, c='g', label = "CVaR, worst 5% of outcomes")
plt.legend(); plt.show()

# Risk exposure and loss
# VaR and risk exposure
# Import the Student's t-distribution
from scipy.stats import t

# Create rolling window parameter list
mu = losses.rolling(30).mean()
sigma = losses.rolling(30).std()
rolling_parameters = [(29, mu[i], s) for i,s in enumerate(sigma)]

# Compute the 99% VaR array using the rolling window parameters
VaR_99 = np.array( [ t.ppf(0.99, *params) 
                    for params in rolling_parameters ] )

# Plot the minimum risk exposure over the 2005-2010 time period
plt.plot(losses.index, 0.01 * VaR_99 * 100000)
plt.show()

#2
# Fit the Student's t distribution to crisis losses
p = t.fit(crisis_losses)

# Compute the VaR_99 for the fitted distribution
VaR_99 = t.ppf(0.99, *p)

# Use the fitted parameters and VaR_99 to compute CVaR_99
tail_loss = t.expect( lambda y: y, args = (p[0],), loc = p[1], scale = p[2], lb = VaR_99 )
CVaR_99 = (1 / (1 - 0.99)) * tail_loss
print(CVaR_99)

# Risk management using VaR & CVaR
# VaR from a fitted distribution
# Visualize the fitted distribution with a plot
x = np.linspace(-0.25,0.25,1000)
plt.plot(x,fitted.evaluate(x))
plt.show()

# Create a random sample of 100,000 observations from the fitted distribution
sample = fitted.resample(100000)

# Compute and display the 95% VaR from the random sample
VaR_95 = np.quantile(sample, 0.95)
print(VaR_95)

# Minimizing CVaR
# Import the EfficientFrontier class
from pypfopt.efficient_frontier import EfficientFrontier

# Import the negative_cvar objective function
from pypfopt.objective_functions import negative_cvar

# Create the efficient frontier instance
ef = EfficientFrontier(None, e_cov)

# Find the cVar-minimizing portfolio weights at the default 95% confidence level
optimal_weights = ef.custom_objective(negative_cvar, returns)

# Display the optimal weights
print(optimal_weights)

# CVaR risk management and the crisis
# Initialize the efficient portfolio dictionary
ef_dict = {}

# For each epoch, assign an efficient frontier instance to ef
for x in ['before', 'during', 'after']: 
    ef_dict[x] = EfficientFrontier(None, e_cov_dict[x])

#2
# Initialize the dictionary of optimal weights
optimal_weights_dict = {}

# Find and display the CVaR-minimizing portfolio weights at the default 95% confidence level
for x in ['before', 'during', 'after']:
    optimal_weights_dict[x] = ef_dict[x].custom_objective(negative_cvar, returns_dict[x])
    
# Compare the CVaR-minimizing weights to the minimum volatility weights for the 'before' epoch
print("CVaR:\n", pd.DataFrame.from_dict(optimal_weights_dict['before']), "\n")
print("Min Vol:\n", pd.DataFrame.from_dict(min_vol_dict['before']), "\n")

# Portfolio hedging: offsetting risk
# Black-Scholes options pricing
# Compute the volatility as the annualized standard deviation of IBM returns
sigma = np.sqrt(252) * IBM_returns.std()

# Compute the Black-Scholes option price for this volatility
value_s = black_scholes(S = 90, X = 80, T = 0.5, r = 0.02, 
                        sigma = sigma, option_type = "call")

# Compute the Black-Scholes option price for twice the volatility
value_2s = black_scholes(S = 90, X = 80, T = 0.5, r = 0.02, 
                         sigma = 2 * sigma, option_type = "call")

# Display and compare both values
print("Option value for sigma: ", value_s, "\n", 
      "Option value for 2 * sigma: ", value_2s)

# Options pricing and the underlying asset
# Select the first 100 observations of IBM data
IBM_spot = IBM[:100]

# Initialize the European put option values array
option_values = np.zeros(IBM_spot.size)

# Iterate through IBM's spot price and compute the option values
for i,S in enumerate(IBM_spot.values):
    option_values[i] = black_scholes(S = S, X = 140, T = 0.5, r = 0.02,
                        sigma = sigma, option_type = "put")

# Display the option values array
option_axis.plot(option_values, color = "red", label = "Put Option")
option_axis.legend(loc = "upper left")
plt.show()

# Using options for hedging
# Compute the annualized standard deviation of `IBM` returns
sigma = np.sqrt(252) * IBM_returns.std()

# Compute the Black-Scholes value at IBM spot price 70
value = black_scholes(S = 70, X = 80, T = 0.5, r = 0.02, 
                      sigma = sigma, option_type = "put")
# Find the delta of the option at IBM spot price 70
delta = bs_delta(S = 70, X = 80, T = 0.5, r = 0.02, 
                 sigma = sigma, option_type = "put")

# Find the option value change when the price of IBM falls to 69.5
value_change = black_scholes(S = 69.5, X = 80, T = 0.5, r = 0.02, 
                             sigma = sigma, option_type = "put") - value

print( (69.5 - 70) + (1/delta) * value_change )

# Chapter 3
# Parametric Estimation
# Parameter estimation: Normal
# Import the Normal distribution and skewness test from scipy.stats
from scipy.stats import norm, anderson

# Fit portfolio losses to the Normal distribution
params = norm.fit(losses)

# Compute the 95% VaR from the fitted distribution, using parameter estimates
VaR_95 = norm.ppf(0.95, *params)
print("VaR_95, Normal distribution: ", VaR_95)

# Test the data for Normality
print("Anderson-Darling test result: ", anderson(losses))

# Parameter estimation: Skewed Normal
# Import the skew-normal distribution and skewness test from scipy.stats
from scipy.stats import skewnorm, skewtest

# Test the data for skewness
print("Skewtest result: ", skewtest(losses))

# Fit the portfolio loss data to the skew-normal distribution
params = skewnorm.fit(losses)

# Compute the 95% VaR from the fitted distribution, using parameter estimates
VaR_95 = skewnorm.ppf(0.95, *params)
print("VaR_95 from skew-normal: ", VaR_95)

# Historical and Monte Carlo Simulation
# Historical Simulation
# Create portfolio returns for the two sub-periods using the list of asset returns
portfolio_returns = np.array([x.dot(weights) for x in asset_returns])

# Derive portfolio losses from portfolio returns
losses = - portfolio_returns

# Find the historical simulated VaR estimates
VaR_95 = [np.quantile(x, 0.95) for x in losses]

# Display the VaR estimates
print("VaR_95, 2005-2006: ", VaR_95[0], '; VaR_95, 2007-2009: ', VaR_95[1])

# Monte Carlo Simulation
# Initialize daily cumulative loss for the 4 assets, across N runs
daily_loss = np.zeros((4,N))

# Create the Monte Carlo simulations for N runs
for n in range(N):
    # Compute simulated path of length total_steps for correlated returns
    correlated_randomness = e_cov @ norm.rvs(size = (4,total_steps))

# Initialize daily cumulative loss for the assets, across N runs
daily_loss = np.zeros((4,N))

# Create the Monte Carlo simulations for N runs
for n in range(N):
    # Compute simulated path of length total_steps for correlated returns
    correlated_randomness = e_cov @ norm.rvs(size = (4,total_steps))
    # Adjust simulated path by total_steps and mean of portfolio losses
    steps = 1/total_steps
    minute_losses = mu * steps + correlated_randomness * np.sqrt(steps)
    daily_loss[:, n] = minute_losses.sum(axis=1)

# Generate the 95% VaR estimate
losses = weights @ daily_loss
print("Monte Carlo VaR_95 estimate: ", np.quantile(losses, 0.95))

# Structural breaks
# Crisis structural break: I
# Create a plot of quarterly minimum portfolio returns
plt.plot(port_q_min, label="Quarterly minimum return")

# Create a plot of quarterly mean volatility
plt.plot(vol_q_mean, label="Quarterly mean volatility")

# Create legend and plot
plt.legend()
plt.show()

# Crisis structural break: II
# Import the statsmodels API to be able to run regressions
import statsmodels.api as sm

# Add a constant to the regression
mort_del = sm.add_constant(mort_del)

# Regress quarterly minimum portfolio returns against mortgage delinquencies
result = sm.OLS(port_q_min, mort_del).fit()

# Retrieve the sum-of-squared residuals
ssr_total = result.ssr
print("Sum-of-squared residuals, 2005-2010: ", ssr_total)

# Crisis structural break: III
# Add intercept constants to each sub-period 'before' and 'after'
before_with_intercept = sm.add_constant(before['mort_del'])
after_with_intercept  = sm.add_constant(after['mort_del'])

# Fit OLS regressions to each sub-period
r_b = sm.OLS(before['returns'], before_with_intercept).fit()
r_a = sm.OLS(after['returns'], after_with_intercept).fit()

# Get sum-of-squared residuals for both regressions
ssr_before = r_b.ssr
ssr_after = r_a.ssr
# Compute and display the Chow test statistic
numerator = ((ssr_total - (ssr_before + ssr_after)) / 2)
denominator = ((ssr_before + ssr_after) / (24 - 4))
print("Chow test statistic: ", numerator / denominator)

# Volatility and extreme values
# Volatility and structural breaks
# Find the time series of returns with and without Citibank
ret_with_citi = prices_with_citi.pct_change().dot(weights_with_citi)
ret_without_citi = prices_without_citi.pct_change().dot(weights_without_citi)

# Find the average 30-day rolling window volatility as the standard deviation
vol_with_citi = ret_with_citi.rolling(30).std().dropna().rename("With Citi")
vol_without_citi = ret_without_citi.rolling(30).std().dropna().rename("Without Citi")

# Combine two volatilities into one Pandas DataFrame
vol = pd.concat([vol_with_citi, vol_without_citi], axis=1)

# Plot volatilities over time
vol.plot().set_ylabel("Losses")
plt.show()

# Extreme values and backtesting
# Compute the 95% VaR on 2009-2010 losses
VaR_95 = np.quantile(estimate_data, 0.95)

# Find backtest_data exceeding the 95% VaR
extreme_values = backtest_data[backtest_data > VaR_95]

# Compare the fraction of extreme values for 2007-2008 to the Var_95 estimate
print("VaR_95: ", VaR_95, "; Backtest: ", len(extreme_values) / len(backtest_data) )

# Plot the extreme values and look for clustering
plt.stem(extreme_values.index, extreme_values)
plt.ylabel("Extreme values > VaR_95"); plt.xlabel("Date")
plt.show()

# Chapter 4
# Extreme value theory
# Block maxima
# Resample the data into weekly blocks
weekly_maxima = losses.resample("W").max()

# Plot the resulting weekly maxima
axis_1.plot(weekly_maxima, label = "Weekly Maxima")
axis_1.legend()
plt.figure("weekly")
plt.show()

#2
# Resample the data into monthly blocks
monthly_maxima = losses.resample("M").max()

# Plot the resulting monthly maxima
axis_2.plot(monthly_maxima, label = "Monthly Maxima")
axis_2.legend()
plt.figure("monthly")
plt.show()

#3
# Resample the data into quarterly blocks
quarterly_maxima = losses.resample("Q").max()

# Plot the resulting quarterly maxima
axis_3.plot(quarterly_maxima, label = "Quarterly Maxima")
axis_3.legend()
plt.figure("quarterly")
plt.show()

# Chapter 4
# Extreme events during the crisis
# Plot the log daily losses of GE over the period 2007-2009
losses.plot()

# Find all daily losses greater than 10%
extreme_losses = losses[losses > 0.10]

# Scatter plot the extreme losses
extreme_losses.plot(style='o')
plt.show()

# Fit extreme distribution to weekly maximum of losses
fitted = genextreme.fit(weekly_max)

# Plot extreme distribution with weekly max losses historgram
x = np.linspace(min(weekly_max), max(weekly_max), 100)
plt.plot(x, genextreme.pdf(x, *fitted))
plt.hist(weekly_max, 50, density = True, alpha = 0.3)
plt.show()

# GEV risk estimation
# Compute the weekly block maxima for GE's stock
weekly_maxima = losses.resample("W").max()

# Fit the GEV distribution to the maxima
p = genextreme.fit(weekly_maxima)

# Compute the 99% VaR (needed for the CVaR computation)
VaR_99 = genextreme.ppf(0.99, *p)

# Compute the 99% CVaR estimate
CVaR_99 = (1 / (1 - 0.99)) * genextreme.expect(lambda x: x, 
           args=(p[0],), loc = p[1], scale = p[2], lb = VaR_99)

# Display the covering loss amount
print("Reserve amount: ", 1000000 * CVaR_99)

# Kernel density estimation
# KDE of a loss distribution
# Generate a fitted T distribution over losses
params = t.fit(losses)

# Generate a Gaussian kernal density estimate over losses
kde = gaussian_kde(losses)

# Add the PDFs of both estimates to a histogram, and display
loss_range = np.linspace(np.min(losses), np.max(losses), 1000)
axis.plot(loss_range, t.pdf(loss_range, *params), label = 'T dist')
axis.plot(loss_range, kde.pdf(loss_range), label = 'Gaussian KDE')
plt.legend(); plt.show()

# CVaR and loss cover selection
# Find the VaR as a quantile of random samples from the distributions
VaR_99_T   = np.quantile(t.rvs(size=1000, *p), 0.99)
VaR_99_KDE = np.quantile(kde.resample(size=1000), 0.99)

# Find the expected tail losses, with lower bounds given by the VaR measures
integral_T   = t.expect(lambda x: x, args = (p[0],), loc = p[1], scale = p[2], lb = VaR_99_T)
integral_KDE = kde.expect(lambda x: x, lb = VaR_99_KDE)

# Create the 99% CVaR estimates
CVaR_99_T   = (1 / (1 - 0.99)) * integral_T
CVaR_99_KDE = (1 / (1 - 0.99)) * integral_KDE

# Display the results
print("99% CVaR for T: ", CVaR_99_T, "; 99% CVaR for KDE: ", CVaR_99_KDE)

# Neural network risk management
# Create the training values from the square root function
y = np.sqrt(x)

# Create the neural network
model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(1))

# Train the network
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(x, y, epochs=100)

## Plot the resulting approximation and the training values
plt.plot(x, y, x, model.predict(x))
plt.show()

# Asset price prediction
# Set the input and output data
training_input = prices.drop('Morgan Stanley', axis=1)
training_output = prices['Morgan Stanley']

# Create and train the neural network with two hidden layers
model = Sequential()
model.add(Dense(16, input_dim=3, activation='sigmoid'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')
model.fit(training_input, training_output, epochs=100)

# Scatter plot of the resulting model prediction
axis.scatter(training_output, model.predict(training_input)); plt.show()

# Real-time risk management
# Create neural network model
model = Sequential()
model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))

# Use the pre-trained model to predict portfolio weights given new asset returns
asset_returns = np.array([0.001060, 0.003832, 0.000726, -0.002787])
asset_returns.shape = (1,4)
print("Predicted minimum volatility portfolio: ", pre_trained_model.predict(asset_returns))

