#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import pandas
from math import sqrt
import scipy.signal


def get_rms(a):
    """
    Return RMS value of an array
    
    Args:
        a: array
    """
    return sqrt(sum(n*n for n in a)/len(a))


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a low-pass filter to data with a Butterworth filter

    Args:
        data:   Data to filter (array-like)
        cutoff: The cutoff frequency of the filter ("-3dB point") (Hz)
        fs:     The sample rate of the data (Hz)
        order:  The order of the filter

    WARNING:
        You must have cutoff <= fs/2 !
    """
    nyq_freq = float(fs)/2.0
    assert cutoff < nyq_freq, "Error: cutoff must be <= fs/2"
    normal_cutoff = float(cutoff) / nyq_freq # normalized cutoff for the butter function
    b, a = scipy.signal.butter(N=order, Wn=normal_cutoff, btype='lowpass',
                        analog=False, output='ba')
    return scipy.signal.filtfilt(b, a, data)


def get_data_fit(x, y, function):
    """Fit data with a function.
    
    Args:
        x,y: pandas Series. y will be fit in function of x.
        function: whether 'inverse', or 'fist_order'.
    
    Returns:
        A dictionary with the following items:
        - fit_func_call: (callable) the fitting function.
        - fit_func_str: (str) the string representation of the fitting function.
        - fit_func_name: (str) the name of the fitting function (same than arg)
        - fit_func_param: (List[float]) parameters of the fitting function.
        - nrmsd: (float) the normalized root mean squared deviation of the fit.
        - disp_inner: (float) the percentage to extend the fitting function to catch 
            50% of data. In other words: 50% of 'y' data is included in the 
            range [ fit(x)-disp*fit(x) , fit(x)+disp*fit(x) ].
        - disp_outer: (float) same principle than disp_inner but to any sample 
            outside of this range are considered as outliers.
    """

    def fit_inverse(_x, _y):
        """Fit the given rising time data with an inverse function.
        
        Args:
            _x,_y: numpy.array-like data. y will be fit in function of x.
        
        Returns:
            (fit_func, fit_func_str)
              fit_func_call: (callable) the fitting function.
              fit_func_str: (str) the string representing the fitting function.
              fit_func_param: (List[float]) parameters found for the function.
        
        Raises:
            ValueError: if data is empty, or if Series have different sizes.
        """
        if len(_x) == 0 or len(_y) == 0:
            raise ValueError("Cannot fit the curve because data is empty !")
        if len(_x) != len(_y):
            raise ValueError("Cannot fit the curve because x and "
                             "y do not have the same size !")
        func = lambda s, a, b: a / s + b  # a and b positive
        popt, pcov = scipy.optimize.curve_fit(
            func, xdata=_x, ydata=_y, bounds=((0, 0), (numpy.inf, numpy.inf)))
        fit_func_call = lambda s: func(s, popt[0], popt[1])
        fit_func_str = 'f(x) = {:.0f} / x  + {:.0f}'.format(popt[0], popt[1])
        return fit_func_call, fit_func_str, popt

    def fit_first_order(_x, _y):
        """Fit the given output speed data with a first order function.
        
        Args:
            _x,_y: numpy.array-like data. y will be fit in function of x.
        
        Returns:
            fit_func, fit_func_str
              fit_func_call: (callable) the fitting function.
              fit_func_str: (str) the string representing the fitting function.
              fit_func_param: (List[float]) parameters found for the function.
            
        Raises:
            ValueError: if data is empty, or if Series have different sizes.
        """
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Cannot fit the curve because data is empty !")
        if len(x) != len(y):
            raise ValueError("Cannot fit the curve because x and "
                             "y do not have the same size !")
        func = lambda s, a, b: a * (1 - numpy.exp(-s / b))  # a and b positive
        popt, pcov = scipy.optimize.curve_fit(
            func,
            xdata=_x,
            ydata=_y,
            bounds=((0, 0.01), (numpy.inf, numpy.inf)))
        fit_func_call = lambda s: func(s, popt[0], popt[1])
        fit_func_str = 'f(x) = {:.0f} * (1 - exp(-x / {:.0f}))'.format(
            popt[0], popt[1])
        return fit_func_call, fit_func_str, popt

    def find_distribution_ranges(_x, _y, fit_func):
        """Find inner and outer limits around fitting curve.
        
        Inner limit is calculated in order to create a range catching 50% of data 
        inside of it. In other words: 50% of 'y' data is included in the range 
        [ fit_func(x) - disp_inner * fit_func(x) , 
          fit_func(x) + disp_inner * fit_func(x) ]
        
        Outer limit is calculated as being the inner limit plus 1.5 times the 
        size of the inner range. It means in the following range:
        [ fit_func(x) - disp_inner * (1+2*1.5) * fit_func(x) , 
        fit_func(x) + disp_inner * (1+2*1.5) * fit_func(x) ]
        
        Args:
            _x,_y: pandas Series.
            fit_func (callable)
        
        Return:
            2-tuple:
              * inner limit dispersion percentage (float)
              * outer limit dispersion percentage (float)
        """
        # Initialize dispersion at 0%
        disp_inner = 0.0
        captured_data_nb = 0
        box_data_nb = int(len(x) * 0.50)  # 50% of data
        fit_series = fit_func(_x)

        while captured_data_nb <= box_data_nb:
            # Compute the number of points which are captured
            # with the current dispersion
            captured_data_nb = 0
            lower_fit_series = fit_series - disp_inner * fit_series
            upper_fit_series = fit_series + disp_inner * fit_series
            is_in_disp = _y.between(lower_fit_series, upper_fit_series)
            captured_data_nb = sum(is_in_disp)
            # Increase dispersion of 1%
            disp_inner += 0.01

        # Compute dispersion for whiskers
        # See details on whiskers of boxplot to understand the
        # following operation.
        disp_outer = disp_inner + 2 * disp_inner * 1.5

        return disp_inner, disp_outer

    # Compute fitting curve
    if function == 'inverse':
        fit_func_call, fit_func_str, fit_func_param = fit_inverse(x, y)
    elif function == 'first_order':
        fit_func_call, fit_func_str, fit_func_param = fit_first_order(x, y)
    else:
        ValueError('Unknown function to fit. Got {}.'.format(function))

    # Compute Normalized root-mean-square deviation
    y_range = y.max() - y.min()
    if y_range == 0:
        # Avoid division by zero
        nrmsd = 0
    else:
        f_y = pandas.Series(fit_func_call(x))
        fit_error = y - f_y
        nrmsd = numpy.sqrt(
                    numpy.sum(
                        numpy.square(
                            fit_error)) \
                / len(y)) \
                / y_range

    # Compute dispersion interval
    disp_inner, disp_outer = find_distribution_ranges(x, y, fit_func_call)

    return {
        'disp_outer': disp_outer,
        'disp_inner': disp_inner,
        'nrmsd': nrmsd,
        'fit_func_str': fit_func_str,
        'fit_func_call': fit_func_call,
        'fit_func_name': function,
        'fit_func_param': fit_func_param
    }
