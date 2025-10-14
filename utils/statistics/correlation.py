import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def compute_returns(df, method='log'):
    """
    Calculate returns from close prices
    
    Parameters:
    - df: DataFrame with 'close' column
    - method: 'log' for log returns, 'simple' for simple returns
    """
    if method == 'log':
        returns = np.log(df['close'] / df['close'].shift(1))
    else:  # simple returns
        returns = df['close'].pct_change()
    
    return returns.dropna()

def compute_serial_correlation(returns, max_lags=20):
    """
    Calculate serial correlation for different lags
    
    Parameters:
    - returns: Series with returns
    - max_lags: Maximum number of lags to calculate
    """
    correlations = []
    lags = range(1, max_lags + 1)
    
    for lag in lags:
        corr = returns.corr(returns.shift(lag))
        correlations.append(corr)
    
    return pd.Series(correlations, index=lags, name='serial_correlation')

def analyze_serial_correlation(dataframes_dict, max_lags=20):
    """
    Analyze serial correlation for multiple dataframes
    
    Parameters:
    - dataframes_dict: Dictionary with names and dataframes
    - max_lags: Maximum number of lags
    """
    results = {}
    
    for name, df in dataframes_dict.items():
        # Calculate returns
        returns = compute_returns(df, method='log')
        
        # Calculate serial correlation
        serial_corr = compute_serial_correlation(returns, max_lags)
        
        # Calculate autocorrelation using statsmodels (includes confidence intervals)
        acf_values, confint = acf(returns, nlags=max_lags, alpha=0.05)
        
        results[name] = {
            'returns': returns,
            'serial_correlation': serial_corr,
            'acf_values': acf_values[1:],  # Exclude lag 0 (always 1)
            'confidence_intervals': confint[1:]
        }
        
        # Display summary statistics
        print(f"\n{name.upper()} DataFrame:")
        print(f"Number of observations: {len(returns)}")
        print(f"Mean returns: {returns.mean():.6f}")
        print(f"Standard deviation: {returns.std():.6f}")
        print(f"\nSerial correlations for first 5 lags:")
        print(serial_corr.head())
    
    return results

def plot_serial_correlations(results, overall_title=None):
    """
    Create plots for serial correlations
    """
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4*len(results)))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        
        # Plot serial correlations
        lags = data['serial_correlation'].index
        correlations = data['serial_correlation'].values
        conf_intervals = data['confidence_intervals']
        
        # Bar plot for correlations
        ax.bar(lags, correlations, alpha=0.7, label='Serial correlation')
        
        # Confidence interval bands
        ax.fill_between(lags, conf_intervals[:, 0], conf_intervals[:, 1], 
                       alpha=0.2, color='gray', label='95% Confidence interval')
        
        # Horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Serial Correlation - {name} bars')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if overall_title:
        fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.show()

def ljung_box_test(returns, lags=10):
    """
    Ljung-Box test for autocorrelation
    H0: No autocorrelation up to specified lag
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    result = acorr_ljungbox(returns, lags=lags, return_df=True)
    return result
