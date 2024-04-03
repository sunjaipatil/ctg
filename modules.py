"""
Main ctg pre-processing sciprt.

It has all the functions for ctg pre-processing

"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.signal import welch

def get_fhr(baby_id: int, file_path = None):
    """
    Input: Baby urn number or file path.
    
    The dataframe will be return with 'FHR' column. 
    
    note: If both file path and baby urn are passed to the function, the file path will be priortised.
    """ 
    # Parent directory 
    
    if not file_path:
        parent_dir = 'ctg/FHR/Moderate_BABYURN/'
        try:
            fhr_path = f'{parent_dir}/{baby_id}.csv'
        except:
            print(f'Please provide the correct parent dir {parent_dir}.The parent directory does not have the file for the URN provided or the parent directory is not correct')
        df = pd.read_csv(fhr_path)
    if file_path:
        df = pd.read_csv(file_path)
    def select_fhr(row):
        """
        MODE 2 is > MODE1 1
        """
        # read mode 2 whenever present else read FHR1
        if row['FHR1MODE'] == 2:
            return row['FHR1']
        elif row['FHR2MODE'] == 2:
            return row['FHR2']
        elif row['FHR3MODE'] == 2:
            return row['FHR3']
        elif row['FHR1MODE'] ==1:
            return row['FHR1']
        elif row['FHR2MODE'] == 1:
            return row['FHR2']
        elif row['FHR3MODE'] == 1:
            return row['FHR3']
        else:
            return row['FHR1']
        
    df['FHR'] = df.apply(select_fhr, axis = 1)
    return df

def remove_type1_outlier(inseries:pd.Series)-> pd.Series:
    """
    Input: panda series (FHR)

    Output: replaces series with np.nan where there is type1 outlier
    """
    outseries = inseries.copy()

    outseries.replace(0, np.nan, inplace = True)


    return outseries

def remove_outliers(inseries:pd.Series, window_size = 480, threshold = 3, threshold_diff = 25, center = True, copy = True):
    """
    Remove outliers with are 3 sigma (threshold) away from the rolling mean and the difference between adjacent is greater 25.
    Input: pandas series (FHR)


    Output: Series which represent where outliers are and the outliers removed series
    """ 
    # Copy the input series
    outseires = inseries.copy()
    non_zero_data = outseires[outseires!=0]
    # replace zero data with mean
    non_zero_data = outseires.replace(0, non_zero_data.mean())
    # calculate rolling mean and std deviation
    rolling_mean = non_zero_data.rolling(window=window_size, center = center).mean()
    rolling_std = non_zero_data.rolling(window=window_size, center = center).std()
    lower_bound = rolling_mean - threshold * rolling_std
    upper_bound = rolling_mean + threshold * rolling_std
    outliers = (non_zero_data < lower_bound) | (non_zero_data > upper_bound)
    
    # Remove those points where the difference between adjacent > 25
    outseires = inseries.copy()
    non_zero_data = outseires[outseires!=0]
    non_zero_data = outseires.replace(0, non_zero_data.mean())
    outliers_diff = non_zero_data.diff()>threshold_diff
    
    # If you don't want a copy of the input series, then replace outseries with inseries in the next two lines
    outseires[outliers_diff == True] = np.nan
    outseires[outliers == True] = np.nan
    if not copy:
        inseries[outliers_diff == True] = 0
        inseries[outliers == True] = 0
        return 
    __import__("IPython").embed()
    return outliers, outliers_diff,outseires

def remove_type2_outliers(inseries:pd.Series, coin: pd.Series, copy = True) -> pd.Series:
    """
    input: provide COIN series along with FHR 
    When maternal heart rate is confused with fetal heart rate, the COIN variable is set to 1 or non-zero

    This is an outlier and should be removed. 
    """
    outseries = inseries.copy()
    outseries[coin!=0] = np.nan
    if not copy:
        inseries[coin!=0] = 0 
        return 
    return outseries

def get_zero_segments(data):
    """
    input: pandas sereies
    
    output: index where the zero (or signal dropout starts) and the length of signal dropout
    """
    data = data.copy()
    # replace all nan with zero
    data = data.replace(np.nan, 0)
    #get all the indicies where data point is zero
    inds = np.where(data==0)[0]
    # Get the difference between consective zero inds
    # If the diff ==1 then they are consective
    diff_inds = np.diff(inds)
    # Get all the inds where difference is greater than 1
    res_inds = np.where(diff_inds!=1)[0]

    # Get the length of zero segments
    zero_segments = {}

    # The first zero ind is going to be inds[0] 
    zero_segments[inds[0]] = inds[res_inds[0]] - inds[0] +1
    # last zero ind
    zero_segments[inds[res_inds[-1]+1]] = inds[-1] - inds[res_inds[-1]+1] +1
    for i, val in enumerate(res_inds[1:]):
        ind1, ind2 = res_inds[i] + 1,val
        length = inds[ind2] -inds[ind1] +1
        zero_segments[inds[ind1]] = length

    return zero_segments

def interpolate_data(inseries, method ='polynomial', order = 3, threshold = 60, copy = True):
    """
    Takes a input series, and interpolates all the zeros which are less than threshold lenght to zero.

    # Note: All outlier removal should be done before. 
    # Note: threshold is 60, which corresponds to 15 seconds at 4Hz sampling frequency. Should be changed according to sampling frequency.
    """

    data = inseries.copy()
    zero_segments = get_zero_segments(data)

    data = data.replace(0, np.nan)
    data = data.interpolate(method = method, order = order, limit_direction = 'both')
   
    for key in zero_segments.keys():
        
        if zero_segments[key]>threshold:
            data[key:key + zero_segments[key]] = np.nan
        else:
            continue
    #if return_zero_segments:
    #   return data, zero_segments
    if not copy:
        inseries = data
        return
    return data




def uc_sensitivity(file_path, urn = None, plotting = None):
    """
    Checks whether there is a peak within the desirable frequency range [0.0033 and 0.0167]
    
    Return - calcuate the percentage area within the range
    
    """
    
    
    data = pd.read_csv(file_path)
    
    uc = data['TOCO']
    if uc.isna().any():
        print("Uterine contractions need to be processed before")
        return
    uc = uc.values
    # Segment size is 16 minutes
    segment_size = 16*60*4 # 4 is the sampling frequency in Hz
    uc_segments = [uc[i:i+segment_size] for i in range(0, len(uc), segment_size)]
    percentage_area_values = np.zeros(len(uc_segments))
    max_frequency_values = np.zeros(len(uc_segments))
    for i,uc_seg in enumerate(uc_segments):
        uc_seg = uc_seg - np.mean(uc_seg)
        sampling_freq = 4
        nfft = int(1.1*len(uc_seg))
        window = 'hann'
        frequencies, psd = welch(uc_seg, fs=sampling_freq, window=window, nfft=nfft)
        # calculate the area
        high_res_freq = np.arange(0,0.0167+0.002 , 0.002)
        high_res_psd = np.interp(high_res_freq, frequencies, psd)
    
        total_area = np.trapz(psd, frequencies)
        indices_within_range = ( high_res_freq>= 0.0033) & (high_res_freq <= 0.0167)
        freq_within_range = high_res_freq[indices_within_range]
        psd_within_range = high_res_psd[indices_within_range]
        area_within_range = np.trapz(psd_within_range, freq_within_range)
        percentage_area_within_range = (area_within_range / total_area) * 100
        percentage_area_values[i] = percentage_area_within_range
        max_frequency_values[i] = frequencies[psd == psd.max()]
       
    #percentage_area_values = sorted(percentage_area_values, reverse = True)
    
    if plotting:
        return frequencies, psd 
    return percentage_area_values, max_frequency_values


def introduce_breaks(inseries, break_len_min = 0, break_len_max = 60):
    """
    This function was used to check the robustness of different interpolation techniques
    
    """
    outseries = inseries.copy()
    integer_array  = np.arange(0, len(inseries))
    start_integers = np.random.choice(integer_array, size = 50, replace = False)
    zero_len = np.random.randint(break_len_min, break_len_max, 50)
    for i, int in enumerate(start_integers):
        outseries[int:int+zero_len[i]] = 0
        
        
    return outseries, start_integers, zero_len

def downsample(inseries: pd.Series, original_frequency= 4, target_frequency = 0.25, copy = False):

    
    downsample_interval = int(original_frequency / target_frequency)
    new_index = inseries.index[::downsample_interval]

    outseries = inseries.groupby(inseries.index // downsample_interval).mean()
    outseries.index = new_index
    if not copy:
        inseries = outseries
        return 

    return outseries 