import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from stepmix import StepMix
from sklearn.model_selection import GridSearchCV

from . import logger

def logit(data: pd.DataFrame, outcome: str, confounders: list, categorical_vars: list = None, 
          dropna: bool = False, show_results: bool = False, forest_plot: bool = False, 
          reference_col: str = None, selected_confounders: list = None, 
          custom_colors: list = None, error_bar_colors: list = None,
          confounder_names: dict = None) -> sm.Logit:
    """
    Fits a logistic regression model to the given data and optionally plots a forest plot of the odds ratios.

    Args:
        data (pd.DataFrame): The input data containing the outcome variable and confounders.
        outcome (str): The name of the outcome variable column.
        confounders (list): A list of confounders column names to be used in the model.
        dropna (bool, optional): Whether to drop rows with missing values. Defaults to False.
        categorical_vars (list, optional): A list of categorical variable column names to be converted to dummy variables. Defaults to None.
        show_results (bool, optional): Whether to print the summary of the logistic regression results. Defaults to False.
        forest_plot (bool, optional): Whether to plot a forest plot of the odds ratios. Defaults to False.
        reference_col (str, optional): The reference column for adjusting odds ratios. Defaults to None.
        selected_confounders (list, optional): A list of selected confounders to be included in the forest plot. Defaults to None.
        custom_colors (list, optional): A list of custom colors for the points in the forest plot. Defaults to None.
        error_bar_colors (list, optional): A list of custom colors for the error bars in the forest plot. Defaults to None.
        confounder_names (dict, optional): A dictionary mapping original confounder names to display names in the forest plot. Defaults to None.

    Returns:
        sm.Logit: The fitted logistic regression model.
    """
    
    # prepare data
    X = data[confounders]
    y = data[outcome]
    logger.info('Prepared data for logistic regression model.')

    # drop rows with missing values
    if dropna:
        initial_length = len(X)
        missing_data = X.isna().sum()
        X = X.dropna()
        y = y.loc[X.index]
        removed_entries = initial_length - len(X)
        logger.info(f'Dropped {removed_entries} rows with missing values.')
        logger.info(f'Columns with most missing values: {missing_data[missing_data > 0].sort_values(ascending=False).head().to_dict()}')

    # convert categorical variables to dummy variables
    if categorical_vars:
        X = pd.get_dummies(X, columns=categorical_vars, drop_first=False)
        logger.info(f'Converted categorical variables to dummy variables: {categorical_vars}')

    # ensure binary variables are integers (0 or 1) instead of boolean
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
        logger.info(f'Converted boolean column {col} to integer.')
    
    # fit logistic regression model
    model = sm.Logit(y, X)
    result = model.fit(disp=0) # disp=0 suppresses output
    logger.info('Fitted logistic regression model.')

    # ORs and 95% CIs
    odds_ratios = np.exp(result.params)
    conf = np.exp(result.conf_int())
    logger.info('Calculated odds ratios and 95% confidence intervals.')

    # DataFrame for plotting
    or_df = pd.DataFrame({
        'OR': odds_ratios,
        'Lower CI': conf[0],
        'Upper CI': conf[1]
    }).reset_index().rename(columns={'index': 'confounder'})

    # exclude the constant term
    or_df = or_df[or_df['confounder'] != 'const']
    logger.info('Excluded the constant term from the odds ratios.')

    # adjust ORs relative to the reference column
    if reference_col and reference_col in or_df['confounder'].values:
        ref_or = or_df.loc[or_df['confounder'] == reference_col, 'OR'].values[0]
        or_df['OR'] /= ref_or
        or_df['Lower CI'] /= ref_or
        or_df['Upper CI'] /= ref_or
        logger.info(f'Adjusted odds ratios relative to reference column: {reference_col}')

    # filter selected confounders for plotting
    if selected_confounders:
        or_df = or_df[or_df['confounder'].isin(selected_confounders)]
        logger.info(f'Selected confounders for plotting: {selected_confounders}')

    # map original confounder names to display names if confounder_names is provided
    if confounder_names:
        or_df['confounder'] = or_df['confounder'].map(confounder_names).fillna(or_df['confounder'])
        logger.info(f'Mapped original confounder names to display names: {confounder_names}')

    # plotting
    if forest_plot:
        plt.figure(figsize=(10, 6))
        
        # set white background
        plt.gca().set_facecolor('white')
        
        # use custom colors (if provided)
        if error_bar_colors is None:
            error_bar_colors = ['grey'] * len(or_df) # default to grey if not provided
            logger.info('Using default grey color for error bars.')
        
        # create error bars for each confounder
        for i in range(len(or_df)):
            # calculate the error margins
            lower_error = or_df['OR'].iloc[i] - or_df['Lower CI'].iloc[i]
            upper_error = or_df['Upper CI'].iloc[i] - or_df['OR'].iloc[i]
            plt.errorbar(or_df['OR'].iloc[i], or_df['confounder'].iloc[i], 
                         xerr=[[lower_error], [upper_error]],
                         fmt='none', color=error_bar_colors[i], capsize=5)
        logger.info('Created error bars for the odds ratios.')

        # use custom colors if provided
        if custom_colors and len(custom_colors) == len(or_df):
            palette = custom_colors
            logger.info(f'Using custom colors for the points: {custom_colors}')
        else:
            palette = ['blue'] * len(or_df)
            logger.info('Using default blue color for the points.')

        # create a scatter plot with custom colors
        sns.scatterplot(data=or_df, x='OR', y='confounder', 
                         color='blue', s=100, zorder=2) # default color for the points
        logger.info('Created scatter plot with default colors.')
        
        if error_bar_colors:
            for i in range(len(or_df)):
                plt.scatter(or_df['OR'].iloc[i], or_df['confounder'].iloc[i], 
                            color=palette[i], s=100, zorder=2) # color for each point
            logger.info('Created scatter plot with custom colors.')

        plt.axvline(x=1, color='gray', linestyle='dotted', zorder=1, linewidth=1)
        logger.info('Added vertical line at OR = 1.')

        plt.title('Odds Ratios with 95% Confidence Intervals')
        plt.xlabel('Odds Ratio')
        plt.ylabel('Confounders')
        plt.grid(False)

        # adjust x-axis ticks
        current_ticks = plt.xticks()[0]
        new_ticks = [tick for tick in current_ticks if tick != 0]
        plt.xticks(new_ticks +  [1]) # include 1 in the x-axis ticks

        # set y-axis limits
        min_x = min(or_df["Lower CI"].min(), or_df["OR"].min()) - 3
        max_x = max(or_df["Upper CI"].max(), or_df["OR"].max()) + 3
        if np.isfinite(min_x) and np.isfinite(max_x):
            plt.xlim(left=min_x, right=max_x)
            logger.info(f'Set x-axis limits: left = {min_x} and right = {max_x}')
        else:
            logger.warning('Could not set x-axis limits as minimum or maximum value is not finite. Using default limits.')

        plt.show()

    # show results
    if show_results:
        print(result.summary())
        logger.info('Finished displaying logistic regression results.')
    
    return result

def lca(data: pd.DataFrame, outcome: str = None, confounders: list = None, 
        n_classes: list = list(range(1,11)), cv: int = 3) -> StepMix:
    """
    Fits a Latent Class Analysis (LCA) model to the given data using StepMix. 
    If no outcome or confounders are provided, an unsupervised approach is used.

    Args:
        data (pd.DataFrame): The input data containing the variables for LCA.
        outcome (str, optional): The name of the outcome variable column. Defaults to None.
        confounders (list, optional): A list of confounders column names to be used in the model. Defaults to None.
        n_classes (list, optional): The number of latent classes to fit. Defaults to a range from 1 to 10.
        cv (int, optional): The number of cross-validation folds for hyperparameter tuning. Defaults to 3.

    Returns:
        model: The fitted LCA model.
    """

    # imply supervised approach if outcome or confounders are provided
    supervised = outcome and confounders

    # prepare data
    if supervised:
        X = data[confounders]
        y = data[outcome]
        logger.info('Prepared data for supervised LCA model.')
    else:
        logger.info('No outcome or confounders provided. Using unsupervised approach.')

    # base model
    model = StepMix(n_components=3, n_steps=1, measurement='bernoulli', structural='bernoulli', random_state=42)

    # hyperparameter tunings
    gs = GridSearchCV(estimator=model, cv=cv, param_grid={'n_components': n_classes})
    if supervised:
        gs.fit(X, y)
    else:
        gs.fit(data)
    logger.info(f'Hyperparameter tuning completed with {n_classes} latent classes and {cv} cross-validation folds.')

    # plot log likelihood
    results = pd.DataFrame(gs.cv_results_)
    results['log_likelihood'] = results['mean_test_score']
    sns.lineplot(data=results, x='param_n_components', y='log_likelihood', marker='o')
    plt.xticks(results['param_n_components'].unique())
    plt.xlabel('Number of Latent Classes')
    plt.ylabel('Log Likelihood')
    plt.show()
    logger.info('Plotted log likelihood against number of latent classes.')

    # best model
    model = gs.best_estimator_
    logger.info(f'Best model selected based on hyperparameter tuning: {model}')

    return model