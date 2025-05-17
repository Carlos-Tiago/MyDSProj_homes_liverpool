import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import BoxStyle
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
import csv
import os
from numba import jit


def is_id_column_name(column_name):
    id_regex = re.compile("(?i)^.+_id$")
    exceptions = ['home_liverpool_id']
    # If exception, say it's not id
    if column_name in exceptions:
        return None
    return re.search(id_regex, column_name)


def is_date_or_time_column_name(column_name):
    # Note: year isn't usually preprocessed (outliers, encoding, etc)
    #  like the other date/time measures, so it's not included
    #
    # add hour, minute, second, etc if ever needed
    date_time_regex = re.compile("^.+_(month|day)$") #cyclics
    return re.search(date_time_regex, column_name)



def is_year_column_name(column_name):
    # Separated from 'is_date_or_time_column_name' since the distinction is
    #  useful during preprocessing
    year_regex = re.compile("^.+_year$")
    return re.search(year_regex, column_name)


def is_month_column_name(column_name):
    # For encoding/decoding months specifically 
    month_regex = re.compile("^.+_month(_(sin|cos))*$")
    return re.search(month_regex, column_name)


def get_max_cardinality_of_months():
    max = 12
    return max


def is_day_column_name(column_name):
    # For encoding/decoding days specifically
    day_regex = re.compile("^.+_day(_(sin|cos))*$")
    return re.search(day_regex, column_name)


def get_max_cardinality_of_days():
    max = 31
    return max


def is_sin_column_name(column_name):
    sin_regex = re.compile("^.+_sin$")
    return re.search(sin_regex, column_name)


def is_cos_column_name(column_name):
    cos_regex = re.compile("^.+_cos$")
    return re.search(cos_regex, column_name)


def is_encoded_column_name(column_name):
    encoded_regex = re.compile("^encoded_.+$")
    return re.search(encoded_regex, column_name)


def split_dataframe_data_by_type(df):
    # Partitions a dataframe and groups them by the folowing types:
    #  Lists
    #  Numerics
    #  Objects (mainly strings, and excluding lists)
    #  Booleans
    
    df_numbers_no_dates_times, df_numbers_dates_times, df_objects, df_bools, df_lists = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 

    for column_name in df.columns:
        row_number = 0
        value = df[column_name].iloc[row_number] 
        # Every data point/ row has a list since there's no NaNs, only empty lists
        if isinstance(value, list):
            df_lists[column_name] = df[column_name].copy(deep=True)
            continue
            
        # Ignore nans before checking type
        while(pd.isna(value)):
            row_number += 1
            value = df[column_name].iloc[row_number]
        
        if np.issubdtype(df[column_name].dtype , np.number) and not is_date_or_time_column_name(column_name):
            # This copy method keeps the indexes. Useful to keep the correspondance
            #  with y_train/test (to delete rows for example)
            df_numbers_no_dates_times[column_name] = df[column_name].copy(deep=True)

        elif np.issubdtype(df[column_name].dtype , np.number):
            df_numbers_dates_times[column_name] = df[column_name].copy(deep=True)
        
        elif np.issubdtype(df[column_name].dtype , object):
            df_objects[column_name] = df[column_name].copy(deep=True)
            
        elif np.issubdtype(df[column_name].dtype , bool): 
            df_bools[column_name] = df[column_name].copy(deep=True) 
    
    return (df_numbers_no_dates_times, df_numbers_dates_times, df_objects,
            df_bools, df_lists) 



def drop_indexes_of_other_data_types(type_using_del_option, *types_to_drop_indexes):
    # Will be returned in the same order in the passed arguments
    types_with_dropped_indexes = [None] * len(types_to_drop_indexes)
    for i, type_to_drop_indexes in enumerate(types_to_drop_indexes):
        if not type_to_drop_indexes.empty:
            type_to_drop_indexes = type_to_drop_indexes.loc[type_using_del_option.index.to_list()]
        types_with_dropped_indexes[i]=type_to_drop_indexes
    
    return types_with_dropped_indexes



def plot_distribution_zscore(X_t, column_name, zscores, mean, standard_deviation, min_nonoutlier_value, max_nonoutlier_value):
    fig, plot1= plt.subplots()         
    plot1 = sns.histplot(X_t[column_name], bins=30, edgecolor='black',
                         kde=True, ax=plot1)
    plot1.set_xlabel(f"{column_name}")
    plot1.set_ylabel(f"Count")
    plot1.set_title(f"{column_name} histogram", y=1.10)

    # Add outlier threshold lines
    #  Done before '.twiny()' so margins are respected
    plot3 = plt.axvline(x=min_nonoutlier_value, ymax=0.5, color='red')
    plot4 = plt.axvline(x=max_nonoutlier_value, ymax=0.5, color='red')
    
    # Compute values corresponding to the standard deviations and add another
    #  x-axis label
    std_devs_values = [mean + (n * standard_deviation) for n in range(zscores[0], zscores[1]+1)]
    plot2 = plot1.twiny() # Share 'y' axis            
    plot2.set_xlabel(f"standard deviations")
    plot2.set_xlim(plot1.get_xlim()) 
    plot2.set_xticks(std_devs_values, labels=range(zscores[0], zscores[1]+1))
    plt.show()
    plot1.figure.savefig(f"zscores_{column_name}")


def plot_distribution_IQR(X_t, column_name):
    plt.figure()
    plt.title(f"{column_name}'s box plot")
    plot=sns.boxplot(data=X_t[column_name])
    plt.show()
    plot.figure.savefig(f"IQR_{column_name}")


def plot_distribution_percentile(X_t, column_name, percentile_threshold, min_nonoutlier_value, max_nonoutlier_value):
    fig, plot1= plt.subplots()         
    plot1 = sns.histplot(X_t[column_name], bins=30, edgecolor='black', kde=True, ax=plot1)
    plot1.set_xlabel(f"{column_name}")
    plot1.set_ylabel(f"Count")
    plot1.set_title(f"{column_name} histogram", y=1.15)

    # Add outlier threshold lines
    #  Done before '.twiny()' so margins are respected
    plot3 = plt.axvline(x=min_nonoutlier_value, ymax=0.5, color='red')
    plot4 = plt.axvline(x=max_nonoutlier_value, ymax=0.5, color='red')
    
    # Compute values corresponding to the standard deviations and
    # add another x-axis label
    plot2 = plot1.twiny() # Share 'y' axis            
    plot2.set_xlabel(f"percentiles")
    plot2.set_xlim(plot1.get_xlim())
    plot2.set_xticks([min_nonoutlier_value, max_nonoutlier_value], 
                     labels=[f"{percentile_threshold*100}th", 
                             f"{(1-percentile_threshold)*100}th"])
    plt.show()
    plot1.figure.savefig(f"percentile_{column_name}")


def get_values_for_scaling_reversion(y_test, y_train):
    y_test_mins = {key:[] for key in y_test.columns}
    y_test_maxs = {key:[] for key in y_test.columns}
    y_train_means = {key:[] for key in y_test.columns}

    for column_name in y_test.columns:

        if np.issubdtype(y_train[column_name].dtype, np.number):
            y_test_mins[column_name] = y_test[column_name].min()
            y_test_maxs[column_name] = y_test[column_name].max()
            y_train_means[column_name] = y_train[column_name].mean() 
            
        else: # Categorical target
            y_test_mins[column_name], y_test_maxs[column_name], y_train_means[column_name] = None, None, None

    return y_test_mins, y_test_maxs, y_train_means


##@jit
def revert_scale(scaled_values, scaler_info, y_test):
    # Convert ndarray back to df so that it can be used in inverse_transform
    scaled_values = pd.DataFrame(scaled_values, columns=y_test.columns, index=y_test.index)
    reverted_values = pd.DataFrame(index=y_test.index)

    scaler = scaler_info[0]
    y_test_min = scaler_info[1]
    y_test_max = scaler_info[2]
    y_train_mean = scaler_info[3]
    
    match(scaler):
        case MinMaxScaler():
            reverted_values = scaler.inverse_transform(scaled_values)
        
        case 'MeanNormalisation':
            for column_name in scaled_values.columns:
                reverted_values[column_name] = scaled_values[column_name] * (y_test_max[column_name] - y_test_min[column_name]) + y_train_mean[column_name]
        
        case StandardScaler():
            reverted_values = scaler.inverse_transform(scaled_values)
        
        case RobustScaler():
            reverted_values = scaler.inverse_transform(scaled_values)
        
        case PowerTransformer(method='yeo-johnson'): 
            # Note: may originate nans
            reverted_values = scaler.inverse_transform(scaled_values) 
            
    
    
    return pd.DataFrame(reverted_values, columns=y_test.columns, index=y_test.index)



def has_numeric_feature(df):
    # Auxiliary function to check if the df needs to go through scaling or not *
    numeric_column_flag = False
    for column_name in df.columns:
        if (np.issubdtype(df[column_name].dtype, np.number)):
            numeric_column_flag = True
            
    return numeric_column_flag


##@jit
def choose_ml_algorithm(y_train, ml_algorithms):
    for column_name in y_train:
        if np.issubdtype(y_train[column_name].dtype, np.number):
            ml_algorithm = ml_algorithms[0]
        elif np.issubdtype(y_train[column_name].dtype, object) or np.issubdtype(y_train[column_name].dtype, bool):
            ml_algorithm = ml_algorithms[1]
        else:
            print(f"ERROR: {column_name} has unforeseen type:"
                  f" {y_train[column_name].dtype}")

    return ml_algorithm


@jit # No real performance difference at the time being
def get_num_extra_digits_numeric_bottleneck(numeric_bottleneck, float32_max):
    extra_digits = 0
    while (numeric_bottleneck > float32_max):
        numeric_bottleneck = numeric_bottleneck / 10
        extra_digits +=1
    
    return extra_digits



def merge_dictionaries(dict1, *other_dicts):
    merged_dict = dict1.copy()
    for other_dict in other_dicts:
        merged_dict.update(other_dict)
    return merged_dict
    

##@jit
def file_append_metrics_and_settings(preprocessing_settings, ml_settings, metrics_results):
    
    metrics_and_settings = merge_dictionaries(ml_settings, metrics_results,
                                              preprocessing_settings)
    key_order = ['target_variable_name'] 
    key_order.extend(metrics_and_settings.keys())
    # Eliminates duplicates. Unlike converting to a set, the dictionary
    #  keeps the order, before converting back to a list.
    key_order = list(dict.fromkeys(key_order)) 
    metrics_and_settings = {key : metrics_and_settings[key] for key in key_order}
    
    # Write header, if file is empty (new)
    with open('performance_results.csv', 'r+', newline='') as csvfile:
        # "Performance results" reader
        p_r_reader = csv.reader(csvfile) 
        try:
            # It seems the first line always "exists", so the pointer needs to
            #  check for two to see if the file is "empty".
            p_r_reader.__next__()
            p_r_reader.__next__()
        except:
            csvfile.seek(0) # Put pointer back at the start
            p_r_writer = csv.writer(csvfile)
            p_r_writer.writerow(metrics_and_settings.keys())       
    
    #Append the settings and performance metrics   
    with open('performance_results.csv', 'a', newline='') as csvfile:
        #* "Performance results" writer
        p_r_dictwriter = csv.DictWriter(csvfile, fieldnames=metrics_and_settings.keys())#*
        p_r_dictwriter.writerow(metrics_and_settings)
         
    return


##@jit
def get_available_settings_options(): 

    #############
    # Settings: #
    #############
    #
    # Catalogue
    #
    # Numeric imputation methods:
    #     DEL - Observation deletion (del. rows) 
    #     1 - Mean imputation
    #     2 - Median imputation
    #     3 - Forward Fill imputation (a kind of Hot Deck imp.)
    #
    #Categorical imputation method
    #     DEL - Observation deletion (del. rows) 
    #     A - Mode imputation (fill with most popular category)
    #
    # Declare distribution type to use the associated detection method  
    #    'normal'-> Z-score
    #    'skewed' -> Interquartile Range (IQR)
    #    'other' -> Percentile (pick max and min percentile threshold)
    # Methods to handle outliers:
    #    0 - Custom (customize the how to handle them regardless. Ignores 
    #         distrib. type!)
    #    1 - Trimming (Removal/Deletion)
    #    2 - Capping
    #
    # Encoding methods (categorical):
    #    1 - One Hot Encoder (add 'drop_method='first' to turn into Dummy Encoder
    #                         and/or solve any multicolinearity issues)
    #    2 - Base N encoding (Default: 16. Add 'base_n=' parameter to the function
    #                        to customize)
    #    3 - Target encoding (Add kfolds=int or smooth=float parameters to the
    #                        function to customize)
    #
    # Scaling methods:
    #    0 - None
    #    1 - Normalisation (Min-max) -> Scale: [0,1]
    #                                   Distribution type: any  
    #                                   Outlier resistance: no 
    #                                   Keeps distribution shape: yes
    #    2 - Normalisation (Mean) -> Centered at: mean
    #                                Distribution type: Gaussian 
    #                                Outlier resistance: some  
    #                                Keeps distribution shape: no
    #    3 - Standardization -> Centered at: mean, 
    #                           Scaled by: stdev=1
    #                           Distribution type: Gaussian  
    #                           Outlier resistance: some  
    #                           Keeps distribution shape: no
    #    4 - Robust Scaling -> Centered at: median, 
    #                          Scaled by: IQR 
    #                          Distribution type: Skewed 
    #                          Outlier resistance: yes  
    #                          Keeps distribution shape: no
    #    5 - Power Transformer (Box-Cox (values>0) / 
    #                           Yeo-Johnson (any values)) ->
    #                          Centered at: 0-mean
    #                          Scaled by: stdev = 1
    #                          Distribution type: not quite Gaussian but
    #                                             suspected to be 
    #                         Outlier resistance: yes(?) 
    #                         Keeps distribution shape: no, makes it Gaussian-like
    #
    # FS/FP Methods (edit parameters of each inside the method):
    #  DR via FS:
    #    1 - Random Forests (Supervised learning)
    #
    #  DR via FP:
    #   (Linear)
    #    2 - PCA
    #    3 - LDA #More suited for classification!
    #    4 - Truncated SVD
    #   (Non-linear)
    #    5 - Kernel PCA (non-linear)
    
    available_settings_options = {
        'target_variable_names' : ['price'],  
        'numeric_imputation_methods_X' : ['DEL', 1, 2, 3],
        'categorical_imputation_methods_X' : ['DEL', 'A'],
        'numeric_imputation_methods_y' : ['DEL', 1, 2, 3],
        'categorical_imputation_methods_y' : ['DEL', 'A'],
        'distribution_types' : ['normal', 'skewed', 'other'],
        'outlier_treating_methods' : [0, 1, 2],
        'encoding_methods' : [1, 2, 3],
        'scaling_methods' : [0, 1, 2, 3, 4, 5],
        'dimensionality_reduction_methods' : [1, 2, 3, 4, 5],
        'ml_algorithms' : [1, 'A']
    }

    return list(available_settings_options.values())


##@jit
def group_to_dictionary(group, preprocessing_settings):
    preprocessing_settings['target_variable_name'] = group[0]
    preprocessing_settings['numeric_imputation_method_X'] = group[1]
    preprocessing_settings['categorical_imputation_method_X'] = group[2]
    preprocessing_settings['numeric_imputation_method_y'] = group[3]
    preprocessing_settings['categorical_imputation_method_y']= group[4]
    preprocessing_settings['distribution_type'] = group[5]
    preprocessing_settings['outlier_treating_method'] = group[6]
    preprocessing_settings['encoding_method'] = group[7]
    preprocessing_settings['scaling_method'] = group[8]
    preprocessing_settings['dimensionality_reduction_method'] = group[9] 
    
    return preprocessing_settings


def get_settings_column_names():
    settings_column_names = [
        'target_variable_name',
        'ml_algorithm',
        'numeric_imputation_method_X',
        'categorical_imputation_method_X',
        'numeric_imputation_method_y',
        'categorical_imputation_method_y',
        'distribution_type',
        'outlier_treating_method',
        'encoding_method',
        'scaling_method',
        'dimensionality_reduction_method'
    ]

    return settings_column_names
