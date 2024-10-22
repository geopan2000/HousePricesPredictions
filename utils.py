from sklearn.preprocessing import LabelEncoder
 
def quality(value):
    'This function converts the quality values to numerical values'
    if value == 'Ex':
        return 5
    elif value == 'Gd':
        return 4
    elif value == 'TA':
        return 3
    elif value == 'Fa':
        return 2
    elif value == 'Po':
        return 1
    else:
        return 0
    
def tranform_label(df, columns):
    'This function converts the categorical values to numerical values'
    for column in columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    return df

# Apply the mapping to both the training and test data
def map_ordinal_features(df, mappings):
    for feature, mapping in mappings.items():
        df[feature] = df[feature].map(mapping)
    return df

def frequency_encoding(train_df, test_df, column):
    # Get the frequency of each category in the train data
    freq_map = train_df[column].value_counts(normalize=True).to_dict()
    
    # Map the frequencies to the train and test data using the same mapping
    train_df[column] = train_df[column].map(freq_map)
    test_df[column] = test_df[column].map(freq_map)
    
    return train_df, test_df

def binary_encoding(df, mappings):
    for feature, mapping in mappings.items():
        df[feature] = df[feature].map(mapping)
    return df