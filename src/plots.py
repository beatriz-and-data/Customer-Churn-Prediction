#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

def eda_map(data, x, y, figsize, feature_bars=None, image=None, target=None):
    """
    Generates a geographical EDA plot with a map and a horizontal bar chart.

    This function creates a figure with two subplots. The left subplot shows a
    scatter plot of data points (e.g., longitude and latitude) on a map image.
    The right subplot displays a horizontal bar chart of the top 10 categories
    of a given feature.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        x (str): The name of the column for the x-axis of the scatter plot (e.g., 'Longitude').
        y (str): The name of the column for the y-axis of the scatter plot (e.g., 'Latitude').
        figsize (tuple): A tuple specifying the width and height of the figure (e.g., (12, 6)).
        feature_bars (str, optional): The name of the categorical column to be plotted in the 
                                      horizontal bar chart (e.g., 'City'). Defaults to None.
        image (str, optional): The file path to the background map image. Defaults to None.
        target (str, optional): The name of the binary target column (e.g., 'Churn'). 
                                If provided, scatter plot points are colored by this target, 
                                and the bar chart is filtered to show counts for target=1. 
                                Defaults to None.

    Shows:
        A matplotlib plot containing the map and the bar chart.
    """
    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Load the background map image
    img_data = mpimg.imread(image) 

    # Define a color palette
    my_palette = sns.color_palette("rocket")
    color_rest = 'lightgray'

    # Assign base colors for non-target and target cases
    color_non_churn_base = my_palette[1] 
    color_churn_base = my_palette[-2]     

    # --- Left Subplot: Map and Scatter Plot ---
    # Display the map image with specified geographical extent
    axes[0].imshow(img_data, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
    # Clean up the map axes
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)

    # If a target variable is provided, color the scatter plot points by it
    if target is not None:
        # Ensure the target column is of 'category' dtype for seaborn
        if not pd.api.types.is_categorical_dtype(data[target]):
            data[target] = data[target].astype('category')

        churn_colors_for_scatter = [color_non_churn_base, color_churn_base] 
        sns.scatterplot(data=data, x=x, y=y, ax=axes[0], hue=target, palette=churn_colors_for_scatter)

    else: # Otherwise, plot all points with a single color
        sns.scatterplot(data=data, x=x, y=y, alpha=0.3, ax=axes[0], color=color_non_churn_base)

    # --- Right Subplot: Horizontal Bar Chart ---
    # Get the top 10 most frequent values from the specified feature
    top_cities = data[feature_bars].value_counts().index[0:10]
    df_filtered = data[data[feature_bars].isin(top_cities)]

    # If a target is provided, filter the data for the bar chart to only include target=1
    if target is not None:
        df_filtered_for_bars = df_filtered[df_filtered[target] == 1]
    else: # Otherwise, use the unfiltered (but top 10) data
        df_filtered_for_bars = df_filtered 

    # Calculate value counts and set the order for the bars
    counts = df_filtered_for_bars[feature_bars].value_counts()
    order = counts.index  

    # Assign colors to bars: highlight the top 5, the rest are gray
    bar_colors = []
    for i, city in enumerate(order):
        if i < 5:
            if target is not None:
                bar_colors.append(color_churn_base) # Highlight color for target
            else:
                bar_colors.append(color_non_churn_base) # Highlight color for non-target
        else:
            bar_colors.append(color_rest) # Gray for the rest

    # Create the horizontal bar plot
    bars = axes[1].barh(order, counts.loc[order], color=bar_colors) 
    axes[1].invert_yaxis() # Display the highest count at the top

    # Clean up the bar chart axes and spines
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].get_xaxis().set_visible(False) 
    axes[1].tick_params(axis='y', length=0) 
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)

    # Annotate each bar with its count
    for bar in bars:
        width = bar.get_width() 
        axes[1].annotate(f'{int(width)}',
                         xy=(width, bar.get_y() + bar.get_height() / 2),
                         xytext=(5, 0),  
                         textcoords="offset points",
                         ha='left', va='center')

    plt.tight_layout()
    plt.show()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_plots(data, features, rows, cols, figsize, plot_type, target=None, target_value=None, scatter_feature=None):
    """
    Generates a grid of various EDA plots like bar charts, histograms, or heatmaps.

    This is a flexible function that can create one of several plot types for a
    list of features, arranged in a grid defined by `rows` and `cols`.

    Args:
        data (pd.DataFrame): The input DataFrame.
        features (list): A list of column names to be plotted.
        rows (int): The number of rows in the subplot grid.
        cols (int): The number of columns in the subplot grid.
        figsize (tuple): The width and height of the entire figure.
        plot_type (str): The type of plot to generate. Options are:
                         'bar': Horizontal bar chart of value counts.
                         'barh': Vertical bar chart of value counts.
                         'hist': Histogram with an optional KDE overlay.
                         'scatter': Scatter plot (requires 'scatter_feature').
                         'corr': Correlation matrix heatmap of the features.
        target (str, optional): The name of a target column for filtering or coloring.
                                Defaults to None.
        target_value (any, optional): A specific value in the `target` column to filter by.
                                      Used with 'bar', 'barh', and 'hist' plots. Defaults to None.
        scatter_feature (str, optional): The column name for the y-axis, required only when
                                         `plot_type` is 'scatter'. Defaults to None.

    Shows:
        A matplotlib plot containing the grid of specified charts.
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Define color palette
    my_palette = sns.color_palette("rocket")
    color_non_churn_base = my_palette[1]
    color_churn_base = my_palette[-2]
    kde_line_color = my_palette[2] 
    churn_scatter_palette = [color_non_churn_base, color_churn_base]

    # Flatten the axes array for easy iteration, handle single plot case
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Loop through the features and create a plot for each
    for i, feature in enumerate(features):
        if i >= len(axes): # Stop if we run out of subplots
            break

        # Ensure target column is 'category' dtype if it exists
        if target is not None and not pd.api.types.is_categorical_dtype(data[target]):
            data[target] = data[target].astype('category')

        # --- Plotting Logic based on plot_type ---

        if plot_type == 'bar': # Horizontal Bar Chart
            if target is not None and target_value is not None:
                df_filtered = data[data[target] == target_value]
                order = df_filtered[feature].value_counts().index
                sns.countplot(y=feature, ax=axes[i], data=df_filtered, legend=False, order=order, color=color_churn_base)
                # Calculation of proportions for filtered bars
                counts = df_filtered[feature].value_counts()
                total_count = counts.sum()
            else:
                order = data[feature].value_counts().index
                sns.countplot(y=feature, ax=axes[i], data=data, order=order, color=color_non_churn_base)
                # Calculation of proportions for bars without a target
                counts = data[feature].value_counts()
                total_count = counts.sum()

            # Annotate bars with percentages
            for p in axes[i].patches:
                count_value = int(p.get_width()) # Count value
                proportion = count_value / total_count * 100 # Proportion in percentage
                x_pos = p.get_width()
                y_pos = p.get_y() + p.get_height() / 2
                # Changed to format as a percentage
                axes[i].text(x_pos + 10, y_pos, f'{proportion:.1f}%', va='center', ha='left', fontsize=9)

            axes[i].get_xaxis().set_visible(False)

        elif plot_type == 'barh': # Vertical Bar Chart
            if target is not None and target_value is not None:
                df_filtered = data[data[target] == target_value]
                order = df_filtered[feature].value_counts().index
                sns.countplot(x=feature, ax=axes[i], data=df_filtered, legend=False, order=order, color=color_churn_base)
                # Calculation of proportions for filtered bars
                counts = df_filtered[feature].value_counts()
                total_count = counts.sum()
            else:
                order = data[feature].value_counts().index
                sns.countplot(x=feature, ax=axes[i], data=data, order=order, color=color_non_churn_base)
                # Calculation of proportions for bars without a target
                counts = data[feature].value_counts()
                total_count = counts.sum()

            # Annotate bars with percentages
            for p in axes[i].patches:
                count_value = int(p.get_height()) # Count value
                proportion = count_value / total_count * 100 # Proportion in percentage
                x_pos = p.get_x() + p.get_width() / 2
                y_pos = p.get_height()
                # Changed to format as a percentage
                axes[i].text(x_pos, y_pos + 5, f'{proportion:.1f}%', ha='center', va='bottom', fontsize=9)

            axes[i].tick_params(axis='x', rotation=60)
            axes[i].get_yaxis().set_visible(False)

        elif plot_type == 'hist': # Histogram
            if target is not None:
                if target_value is None: # Hue by target
                    sns.histplot(x=feature, hue=target, ax=axes[i], data=data, legend=False, edgecolor=None, kde=True, line_kws={'color': kde_line_color})
                else: # Filtered by target value
                    df_filtered = data[data[target] == target_value]
                    sns.histplot(x=feature, ax=axes[i], data=df_filtered, legend=False, color=color_churn_base, edgecolor=None, kde=True, line_kws={'color': kde_line_color})
            else: # Simple histogram
                sns.histplot(x=feature, ax=axes[i], data=data, color=color_non_churn_base, edgecolor=None, kde=True, line_kws={'color': kde_line_color})

            axes[i].tick_params(axis='y', labelsize=8) 
            axes[i].tick_params(axis='x', labelsize=8)

        elif plot_type == 'scatter': # Scatter Plot
            if target is not None and scatter_feature is not None:
                sns.scatterplot(x=feature, y=scatter_feature, hue=target, ax=axes[i], data=data, palette=churn_scatter_palette, legend='full')
            elif scatter_feature is not None:
                sns.scatterplot(x=feature, y=scatter_feature, ax=axes[i], data=data, color=color_non_churn_base)
            else:
                # English translation of the original Portuguese print statement
                print(f"For plot_type='scatter', 'scatter_feature' must be provided.")

        elif plot_type == 'corr': # Correlation Matrix
            if i == 0: # Only plot on the first subplot
                df_corr = data[features].corr(numeric_only=True)
                sns.heatmap(df_corr, vmin=-1, vmax=1, annot=True, cmap='flare', ax=axes[i])
                # English translation of the original Portuguese title
                axes[i].set_title('Correlation Matrix', fontsize=12)
            else: # Turn off other subplots
                axes[i].axis('off')

        # --- General Styling for different plot types ---

        if plot_type in ['bar', 'barh']:
            axes[i].set_title(feature)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].grid(False)

        if plot_type in ['hist', 'scatter']:
            axes[i].set_xlabel(feature, fontsize = 8)
            axes[i].set_ylabel("")
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].grid(False)

    # Turn off any unused subplots in the grid
    for j in range(len(features), len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


# In[ ]:




