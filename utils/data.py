#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy
from tqdm import tqdm_notebook
import yaml
import pickle
from . import signal as sigutils


def store_op_data(file_path, data_type, system_name, processing_results, op_key,
                  **kwargs):
    """Store data of an operating point in a pickle file.
    
    Args:
        file_path: (str) Path to the file to write.
        data_type: (str) Type of the data, 'sine' or 'step'
        system_name: (str) Name of the system for this data.
        processing_results: (List[ProcessingResult] from utils.signal.control)
        op_key: 3-tuple containing the operating point description:
            * 2-tuple for the current limits
            * 2-tuple for the temperature limits
            * load direction
    
    Returns:
        None
    """
    # Initialize the dictionary to store in the pickle file
    stored_dict = {
        'data_type': data_type,
        'processing_results': processing_results,
        'operating_point': {
            'system_name': system_name,
            'current_range': op_key[0],
            'temperature_range': op_key[1],
            'load_direction': op_key[2],
        }
    }

    # Store kwargs
    stored_dict.update(kwargs)

    if not file_path.endswith('.pickle'):
        file_path = file_path + '.pickle'

    with open(file_path, 'wb') as f:
        pickle.dump(stored_dict, f)


def load_op_data(file_path):
    """Load data of an operating point from a pickle file.
    
    Args:
        file_path: (str) Path to the pickle file
    
    Returns:
        The object contained in the pickle file, which a
        dictionnary in this case.
    """
    if not file_path.endswith('.pickle'):
        file_path = file_path + '.pickle'

    if not os.path.isfile(file_path):
        raise FileNotFoundError('{} does not exist.'.format(file_path))

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


def get_all_processed_data(csv_paths, system_name_override=None,
                           filter_bad_steps=True):
    """Get all the processing results for steps or sines from the given CSV files.
    
    Args:
        csv_paths: (List[str]) paths to CSV files containing the step or sine signals.
            It is determined if it is a step file or a sine file by searching
            for 'step' or 'sine' in the filename.
        system_name_override: (str) New name of the system which will override the 
            one written in the metadata file.
        filter_bad_steps: (bool) set to True (default) to discard steps that
            have inconsistencies and should not be taken into account. To know
            more about this filter, read signal.control.process_steps_signal
            This arguments has an effect on steps only.
    
    Returns:
        A list of ProcessingResult objects.
    """

    # Initialize the empty list that will contain the ProcessingResult
    # objects
    all_data = []

    # For each CSV file:
    #  * Get metadata of this measurement
    #  * Process data to get KPIs
    #  * Add results to the final dictionary of results
    for csv_path in tqdm_notebook(csv_paths, desc='CSV files', leave=False):
        # For each .csv file, there should be a file containing metadata
        metadata_path = os.path.splitext(csv_path)[0] + '.yaml'
        if os.path.isfile(metadata_path):
            with open(metadata_path, 'r') as stream:
                full_metadata = yaml.safe_load(stream)
            metadata = full_metadata['measurement_info']
            if system_name_override:
                metadata['system_name'] = system_name_override
        else:
            raise FileNotFoundError("Cannot find metadata file {}".format(metadata_path))
        # Process the signal
        if '_step_' in csv_path:
            new_data = sigutils.control.process_steps_signal(csv_path,
                                                             metadata,
                                                             filter_bad_steps)
        elif '_sine_' in csv_path:
            new_data = [sigutils.control.process_sine_signal(csv_path, metadata)]
        else:
            raise RuntimeError("Cannot find _step_ or _sine_ in the filename: {}".format(csv_path))
        # Store the results
        all_data.extend(new_data)

    return all_data


def get_operating_groups(processing_results,
                         current_bins,
                         temperature_bins,
                         min_size=50):
    """Split results into groups, one for each operating point.

    Results will be split into groups in function of:
        * Steady state current
        * Temperature
        * Load direction

    Load direction MUST be in the metadata attribute of each step,
    otherwise it will take the default value numpy.nan.

    Args:
        processing_results: List of ProcessingResult objects.
        current_bins: (float or List[(float,float)]) If a single float,
            ranges of current values will be made automatically with this
            float as the size of the range. If a list of 2-tuples, each tuple
            is a range of current. For instance, if current_bins =
            [(0, 1), (1, 1.5)], then first range is [0A, 1A[ and second range
            is [1A, 1.5[. Note that second value is excluded from the range.
            Ranges are allowed to overlap.
        temperature_bins: Same description than for current, but for
            temperature.
        min_size: (int) (optional) the minimum number of steps to keep the
            operating group. Default is 50.

    Return:
        (Dict[List[int]], List)
          - A dictionary where keys are tuples describing the operating point:
            ((Min current, Max current), (Min temperature, Max temperature),
            Torque relative direction)
            Each item of the dictionary is a list of indexes from
            processing_results to indicate which result is in each group.
          - A list of indexes being uncategorisable, no present in any
            operating point.
    """
    # Check content of current_bins and temperature_bins
    for bins in [current_bins, temperature_bins]:
        if isinstance(bins, (float, int)):
            if bins <= 0:
                raise ValueError("bins must be a stricly positive float")
        elif isinstance(bins, list):
            for _bin in bins:
                if len(_bin) != 2:
                    raise ValueError("bins must contain 2-tuples.")
                if (not isinstance(_bin[0], (float, int)) or
                    not isinstance(_bin[1], (float, int))):
                    raise ValueError("bins must contain tuples of strictly "
                                     "positive floats.")
                if _bin[0] >= _bin[1]:
                    raise ValueError("bins must contain 2-tuples of float "
                                     "where the first float is lower than the "
                                     "second.")
        else:
            raise ValueError("bins must be either a number or a list of "
                             "2-tuples")

    ret = {}
    uncategorisable = []

    for res_id, res_obj in tqdm_notebook(
            enumerate(processing_results), desc='Results', leave=False):

        # Get parameters to find the operating point
        # For steps
        if res_obj.metadata['data_type'] == 'step':
            current = res_obj.kpis['SteadyStateCurrent']
            temp = res_obj.kpis['Temperature']

            # Default load direction is numpy.nan
            if ('load_direction' in res_obj.metadata
                and res_obj.metadata['load_direction'] is not None):
                dirload = res_obj.metadata['load_direction']
            else:
                dirload = numpy.nan

        # For sine waves
        elif res_obj.metadata['data_type'] == 'sine':
            current = res_obj.kpis['MeanCurrent']
            temp = res_obj.kpis['MeanTemperature']
            dirload = 0  # default for sine

        # Do not consider NaN values
        if current is numpy.nan or dirload is numpy.nan:
            uncategorisable.append(res_id)
            continue

        # Place the step in the right current and temperature ranges
        # As bins can overlap, this step could belong to several ones
        current_bin_set = set()
        temperature_bin_set = set()
        # The following lambda is used to compute bins when a single float is
        # given as argument
        get_bin_range = lambda val, bin_size: (
            abs(bin_size * (val // bin_size)),
            abs(bin_size * (val // bin_size) + bin_size)) # yapf: disable
        if isinstance(current_bins, (float, int)):
            # Compute the bin range of current and temperature,
            # based on given bin sizes
            current_bin_set.add(get_bin_range(current, current_bins))
        else:
            # bins are directly given as arguments
            for current_bin in current_bins:
                if current_bin[0] <= current < current_bin[1]:
                    current_bin_set.add(current_bin)
        # do the same for temperature
        if isinstance(temperature_bins, (float, int)):
            temperature_bin_set.add(get_bin_range(temp, temperature_bins))
        else:
            for temperature_bin in temperature_bins:
                if temperature_bin[0] <= temp < temperature_bin[1]:
                    temperature_bin_set.add(temperature_bin)

        # For each combination of ranges,
        # Make the key tuple of the corresponding operating point
        # And add the ID of the current result to the list of IDs
        # of this operating point
        # If it does not belong to any range, just skip
        if len(current_bin_set) == 0 or len(temperature_bin_set) == 0:
            continue
        for current_bin in current_bin_set:
            for temperature_bin in temperature_bin_set:
                op_point = (current_bin, temperature_bin, dirload)
                if op_point not in ret:
                    ret[op_point] = []
                ret[op_point].append(res_id)

    # Remove operating groups that are too small
    for key, ids in list(ret.items()):
        if len(ids) < min_size:
            ret.pop(key)

    return ret, uncategorisable
