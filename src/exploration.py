###Functions to Help Explore the Pitch Data###
def get_pitch_types_by_year(pitcher_data, date_col = None, use_gameday = True):
    '''Given DF for a single pitcher, returns a DF of their pitch type counts by year
    
    pitcher_data: DF of pitch data for a single pitcher
    date_col: if present, the string name of the column that contains the date on which the pitch was thrown
    use_gameday: whether or not the gameday_link column should be used to derive the date'''
    
    #extract the date from the gameday_id, if necessary
    if use_gameday:
        pitcher_data['date'] = pitcher_data['gameday_link'].str.slice(start = 4, stop = 14)
        pitcher_data['date'] = pitcher_data['date'].str.replace("_", "-")
        pitcher_data['date'] = pd.to_datetime(pitcher_data['date'])
        
    #Check for date col and reassign
    if date_col is not None:
        pitcher['date'] = pitcher[date_col]
    
    #Index on the date and aggregate to the year (season) level
    pitcher_data = pitcher_data.set_index('date')
    pitcher_data = pitcher_data.groupby([lambda x: x.year, 'pitch_type']).size()
    
    #Unstack the data so each pitch type can be plotted
    unstacked = pitcher_data.unstack()
    return unstacked

def plot_pitch_types(pitcher_data, pitcher_name = 'unspecified'):
    '''Thus function takes in pitcher data (like the verlander sample), and creates a plot of pitch types over time.
    Variable descriptions:
    
    pitcher_data: Pandas DF for a single pitcher
    pitcher_name: Name of the pitcher's data being passed into the function'''

    #plot
    pitcher_data.plot(title = 'Pitches over time for ' + pitcher_name)

def find_anomalous_pitching_behavior(pitcher_data, min_pitches = 100, pitcher_id = 'pitcher_name', std_dev_thresh = 2, perc_thresh = 20):
    '''This function uses a control charts methodology to loop through each year of a pitchers history
    and flag any years where a pitch count is more than a certain number of standard deviations away from the mean.
    Obviously, we can only start this after 2 years of data, so those first two years simply look at differences
    between pitch percentages.'''
    
    #Run through all the pitchers in the data set and flag them as potentially anomalous with reasons
    pitchers = pitcher_data[pitcher_id].unique()
    anomalous_pitcher_dict = {}
    
    for pitcher in pitchers:
        
        #Aggregate to the yearly level, drop all rows with only NaNs, and fill NaNs with 0s
        pitch_data = get_pitch_types_by_year(pitcher_data[pitcher_data[pitcher_id] == pitcher])
        pitch_data = pitch_data.dropna(thresh = 1, axis = 1)
        pitch_data = pitch_data.fillna(0)
        
        ###Handling the first two years###
        years = pitch_data.index
        subset = pitch_data.loc[pitch_data.index <= years[1]]

        #Get pitches as a percent of total pitches thrown that year
        subset_perc = subset.apply(lambda x: x / x.sum() * 100, axis = 1)

        #Calculate the absolute percentage differences between the two years
        subset_perc_diff = subset_perc.diff().irow(1).abs()
        pitch_count_sums = subset.sum()

        #Loop through each pitch and check it for criteria
        for pitch in subset.columns:

            #check if pitch minimum met
            if pitch_count_sums[pitch] > (min_pitches * 2):

                #Check if the absolute difference in pitch percentages is greater than threshold
                if subset_perc_diff[pitch] > perc_thresh:

                    #Check for previous entry in dictionary
                    if pitcher not in anomalous_pitcher_dict.keys():

                        anomalous_pitcher_dict[pitcher] = []

                    #Add the data for the pitcher
                    anomalous_pitcher_dict[pitcher] += [(years[1], pitch)]

        #Handling subsequent years if there are more
        if len(years) > 2:
            for year in years[2:len(years)]:

                #Grab data from all previous years for calculating mean and std_dev and loop through each pitch
                subset = pitch_data.loc[pitch_data.index < year]
                for pitch in subset.columns:

                    #Check if there's enough pitches in the year to be considered significant
                    cur_yr_pitch_total = pitch_data[pitch].loc[year]
                    if cur_yr_pitch_total >= min_pitches:

                        #Calculate the z_score for the current pitch total
                        sd = subset[pitch].std()
                        average = subset[pitch].mean()
                        cur_z_score = (cur_yr_pitch_total - average) / sd

                        #Check to see if the new value is out of bounds
                        if cur_z_score > std_dev_thresh:

                            #Check for previous entry in dictionary
                            if pitcher not in anomalous_pitcher_dict.keys():
                                anomalous_pitcher_dict[pitcher] = []

                            #Add the data for the pitcher
                            anomalous_pitcher_dict[pitcher] += [(year, pitch, cur_z_score)]
                            
    return anomalous_pitcher_dict
	
def get_date_from_gameday_id(pitch_df):
    '''Function to extract the pitch date from the "gameday_link" column of a pitch DF
    Input:
        pitch_df: Pandas DF containing the column "gameday_link"
    Output:
        Same Pandas dataframe as input except that it now contains a new Pandas datetime column in the format "yyyy-mm-dd"'''

    pitch_df['date'] = pitch_df['gameday_link'].str.slice(start = 4, stop = 14)
    pitch_df['date'] = pitch_df['date'].str.replace("_", "-")
    pitch_df['date'] = pd.to_datetime(pitch_df['date'])

    return pitch_df
	
def split_test_train(pitcher_df, date, date_col = 'date'):
    '''Takes in a pandas df of pitcher data (one or more pitchers) and splits it into testing and training features and targets.
    It also splits Categorical variables up and binarizes them as their own columns
    Input Args:
        pitcher_df: Pandas dataframe containing all pitch data for a single pitcher
        date: string in the form yyyy-mm-dd, specifying the cutoff for splitting test/train
    Output:
        Dictionary containing:
            train_data: Pandas feature DF for training data
            train_targets: Pandas Series of training data targets (pitch_type)
            test_data: Pandas feature DF for testing data
            test_targets: Pandas Series of testing data targets (pitch_type)'''
    
    #Reshaping
    from pandas.core.reshape import get_dummies #Note: requires Pandas 0.16 +
    pitcher_subset = pitcher_df.drop('pitch_type', axis = 1)
    pitcher_subset = get_dummies(pitcher_subset)
    
    #split the data and store it in a dictionary
    pitcher_dict = {}
    pitcher_dict['train_data'] = pitcher_subset[pitcher_subset[date_col] < date].drop(date_col, axis = 1)
    pitcher_dict['train_targets'] = pitcher_df['pitch_type'][pitcher_df[date_col] < date].astype('category')
    pitcher_dict['test_data'] = pitcher_subset[pitcher_subset[date_col] >= date].drop(date_col, axis = 1)
    pitcher_dict['test_targets'] = pitcher_df['pitch_type'][pitcher_df[date_col] >= date].astype('category')
    
    return pitcher_dict