import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import statsmodels.api as sm
from stepmix import StepMix
from sklearn.model_selection import GridSearchCV

from . import logger

def logit(data: pd.DataFrame, outcome: str, confounders: list, categorical_vars: list = None, 
          dropna: bool = False, show_results: bool = False, forest_plot: bool = False, 
          reference_col: str = None, selected_confounders: list = None, 
          custom_colors: list = None, error_bar_colors: list = None) -> sm.Logit:
    """
    Fits a `statsmodels <https://www.statsmodels.org/stable/index.html>`_ logistic regression model to the given data and optionally plots a forest plot of the odds ratios.

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

    Returns:
        sm.Logit: The fitted logistic regression model.

    Examples:
        >>> import pandas as pd
        >>> from utils.analysis import logit
        >>> data = pd.DataFrame({
        ...     'outcome': [1, 0, 1, 0, 1],
        ...     'confounder_1': [5, 3, 6, 2, 7],
        ...     'confounder_2': [1, 0, 1, 0, 1]
        ... })
        >>> result = logit(
        ...     data=data, 
        ...     outcome='outcome', 
        ...     confounders=['confounder_1', 'confounder_2'], 
        ...     forest_plot=True, 
        ...     reference_col='confounder_1'
        ... )
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

        # create a scatter plot
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
        n_classes: list = list(range(1, 11)), fixed_n_classes: int = None, cv: int = 3, 
        assignments: bool = False, polar_plot: bool = False, cmap: str = 'tab10') -> StepMix:
    """
    Fits a Latent Class Analysis (LCA) model to the given data using `StepMix <https://stepmix.readthedocs.io/en/latest/api.html#stepmix>`_. 
    If no outcome or confounders are provided, an unsupervised approach is used.

    Args:
        data (pd.DataFrame): The input data containing the variables for LCA.
        outcome (str, optional): The name of the outcome variable column. Defaults to None.
        confounders (list, optional): A list of confounders column names to be used in the model. Defaults to None.
        n_classes (list, optional): The number of latent classes to fit. Defaults to a range from 1 to 10.
        fixed_n_classes (int, optional): A fixed number of latent classes to use instead of tuning. Defaults to None.
        cv (int, optional): The number of cross-validation folds for hyperparameter tuning. Defaults to 3.
        assignments (bool, optional): Whether to return the latent class assignments for the observations. Defaults to False.
        polar_plot (bool, optional): Whether to plot a polar plot of the latent class assignments. Defaults to False.
        cmap (str, optional): The colormap to use for plotting clusters. Defaults to 'tab10'.

    Returns:
        StepMix: The fitted LCA model. If `assignments` is True, returns a tuple of (model, data_updated), where:
            - model (StepMix): The fitted LCA model.
            - data_updated (pd.DataFrame): The original data with an additional column for latent class assignments.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from utils.analysis import lca
        >>> data = pd.DataFrame({
        ...     'var_1': np.random.randint(0, 2, 100),
        ...     'var_2': np.random.randint(1, 10, 100),
        ...     'var_3': np.random.randint(1, 5, 100),
        ...     'var_4': np.random.randint(0, 2, 100),
        ...     'var_5': np.random.randint(1, 10, 100),
        ...     'var_6': np.random.randint(1, 5, 100)
        ... })
        >>> model = lca(data=data, n_classes=[2, 3, 4, 5])
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
    base_model = StepMix(n_components=3, n_steps=1, measurement='bernoulli', structural='bernoulli', random_state=42)

    # hyperparameter tuning or fixed model fitting
    if fixed_n_classes is not None:
        logger.info(f'Using fixed number of latent classes: {fixed_n_classes}.')
        model = StepMix(n_components=fixed_n_classes, n_steps=1, measurement='bernoulli', structural='bernoulli', random_state=42)
        if supervised:
            model.fit(X, y)
        else:
            model.fit(data)
    else:
        gs = GridSearchCV(estimator=base_model, cv=cv, param_grid={'n_components': n_classes})
        if supervised:
            gs.fit(X, y)
        else:
            gs.fit(data)
        logger.info(f'Hyperparameter tuning completed with {n_classes} latent classes and {cv} cross-validation folds.')
        model = gs.best_estimator_
        logger.info(f'Best model selected based on hyperparameter tuning: {model}')

    # plot log likelihood (if not using fixed classes)
    if fixed_n_classes is None:
        results = pd.DataFrame(gs.cv_results_)
        results['log_likelihood'] = results['mean_test_score']
        sns.lineplot(data=results, x='param_n_components', y='log_likelihood', marker='o')
        plt.xticks(results['param_n_components'].unique())
        plt.xlabel('Number of Latent Classes')
        plt.ylabel('Log Likelihood')
        plt.show()
        logger.info('Plotted log likelihood against number of latent classes.')

    # predict latent class assignments
    if assignments or polar_plot:

        # copy data to avoid modifying the original
        data_updated = data.copy()

        # predict latent class assignments
        if supervised:
            predictions = model.predict(X, y)
        else:
            predictions = model.predict(data)
        logger.info(f'Predicted {len(predictions)} latent class assignments.')
        
        # add latent class assignments (starting from 1)
        data_updated['latent_class'] = predictions + 1
        logger.info('Merged latent class assignments with observations.')

    # plot polar plot
    if polar_plot:

        # use all columns as confounders if not provided
        if not supervised:
            confounders = data.columns.tolist()
            logger.info('Using all columns as confounders for the polar plot in unsupervised LCA model.')

        # calculate prevalence of each latent class
        class_prevalences = data_updated.groupby('latent_class')[confounders].mean().reset_index()
        total_prevalences = data_updated[confounders].mean()
        
        # normalize prevalences
        normalized_prevalences = class_prevalences.copy()
        for confounder in confounders:
            normalized_prevalences[confounder] = class_prevalences[confounder] / total_prevalences[confounder]
        
        logger.info('Calculated normalized prevalence for each confounder in each latent class.')

        # assign latent classes to confounders
        assigned_classes = {}
        for confounder in confounders:

            # get the class with the highest value (if not empty)
            max_value = normalized_prevalences[confounder].max()
            max_classes = normalized_prevalences[normalized_prevalences[confounder] == max_value]['latent_class'].values
            
            # assign class with highest value
            if max_classes.size == 0:
                logger.warning(f'Confounder {confounder} has no classes assigned.')
                assigned_classes[confounder] = None
            else:
                if max_classes.size > 1:
                    logger.warning(f'Confounder {confounder} has multiple classes with the same normalized prevalence. Choosing the first one.')
                max_class = max_classes[0]
                assigned_classes[confounder] = max_class
                logger.info(f'Assigned latent class {max_class} to confounder {confounder} with normalized prevalence {max_value:.4f}.')

        # plot polar plot
        fig = go.Figure()
        latent_classes = data_updated['latent_class'].unique()
        colors = sns.color_palette(cmap, n_colors=len(latent_classes)).as_hex()
        for i, latent_class in enumerate(sorted(latent_classes)):

            # filter data for the latent class
            class_data = normalized_prevalences[normalized_prevalences['latent_class'] == latent_class]
            class_values = class_data[confounders].values.flatten()

            # skip if no data available
            if class_data.empty:
                logger.warning(f'No data available for latent class {latent_class}. Skipping.')
                continue
            
            # plot polar plot
            fig.add_trace(go.Scatterpolar(
                r=class_values.tolist() + [class_values[0]], # close the shape
                theta=confounders + [confounders[0]], # close the shape
                name=f'Latent Class {latent_class}', # name for legend
                fill='toself', # fill area inside the shape
                fillcolor=f'rgba({int(int(colors[i][1:3], 16))}, {int(int(colors[i][3:5], 16))}, {int(int(colors[i][5:7], 16))}, 0.1)', # fill color (with transparency)
                line=dict(color=colors[i]), # color for the line
            ))
            logger.info(f'Added polar plot for latent class {latent_class}.')

        # update layout and show figure
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, showline=True, linecolor='rgba(0,0,0,0.1)', gridcolor='rgba(0,0,0,0.1)'),
                angularaxis=dict(tickfont=dict(size=24), linecolor='grey', gridcolor='rgba(0,0,0,0.1)'),
                bgcolor='white'
            ),
            showlegend=True,
            legend=dict(font=dict(size=24)),
            paper_bgcolor='rgba(255,255,255)',
            plot_bgcolor='rgba(255,255,255)'
        )
        fig.show()

    # return based on parameters
    if assignments:
        return model, data_updated
    else:
        return model