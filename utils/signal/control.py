#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import namedtuple
import copy
import numpy
import pandas
from scipy.signal import find_peaks
from .generic import get_rms, butter_lowpass_filter, get_data_fit


ProcessingResult = namedtuple("ProcessingResult", "dataframe metadata kpis")


def process_sine_signal(csv_path, metadata, filter_signals=False):
    """Process a .csv file to get motor control KPIs on sine.

    Args:
        csv_path (str): Path to the .csv file to process.
          It must contain Position, Current, Temperature, of a joint which is
            doing a sine. Time must be in seconds.
        metadata (dict): metadata for this csv file. Must contain the following keys:
          * joint_name (str): the name of the joint
          * posture_name (str): the name of the initial posture of the robot
          * body_id (str): the body ID of the tested robot
        filter_signals (bool): True to enable the filtering of the signals.

    Returns:
        (ProcessingResult) A single ProcessingResult object.
        ProcessingResult contains 3 attributes:
            * dataframe: a pandas dataframe containing the raw signals.
            * metadata: a dictionary of metadata.
            * kpis: a dictionary of computed KPIs.
        
    """
    # --------------------------------------------------------------------------
    # VARIABLES INITIALIZATION
    # --------------------------------------------------------------------------
    metadata['data_type'] = 'sine'

    joint_name = metadata['system_name']

    # Initialize the dictionary of KPI that will be returned
    kpis = {}

    # --------------------------------------------------------------------------
    # FILES INITIALIZATION
    # --------------------------------------------------------------------------
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("Cannot find file at {}".format(csv_path))

    # Load the CSV into the main dataframe
    main_dataframe = pandas.read_csv(filepath_or_buffer=csv_path)

    # --------------------------------------------------------------------------
    # DATA PROCESSING
    # --------------------------------------------------------------------------
    pos_act_col = joint_name + "PositionActuatorValue"
    pos_sen_col = joint_name + "PositionSensorValue"

    # If they are not there, compute extra-signals that are not retrieved from
    # the robot.
    extra_cols = ("TimeDispersion", joint_name + "CommandSpeed",
                  joint_name + "CommandAcceleration",
                  joint_name + "SensorSpeed", joint_name + "SensorAcceleration")

    # Check if extra cols are already there before creating them
    if not all([c in main_dataframe.columns for c in extra_cols]):
        # Compute time dispersion
        main_dataframe["TimeDispersion"] = main_dataframe["Time"].diff(1)
        time_disp_sec = main_dataframe["TimeDispersion"]
        # Compute command speed
        main_dataframe[joint_name + "CommandSpeed"] = (
            main_dataframe[pos_act_col].diff(1) / time_disp_sec)
        # Compute command acceleration
        main_dataframe[joint_name + "CommandAcceleration"] = main_dataframe[
            joint_name +
            "CommandSpeed"].diff(1) / main_dataframe["TimeDispersion"]
        # Compute sensor speed
        main_dataframe[
            joint_name +
            "SensorSpeed"] = main_dataframe[pos_sen_col].diff(1) / time_disp_sec
        # Compute sensor acceleration
        main_dataframe[joint_name + "SensorAcceleration"] = main_dataframe[
            joint_name +
            "SensorSpeed"].diff(1) / main_dataframe["TimeDispersion"]
    else:
        time_disp_sec = main_dataframe["TimeDispersion"]

    sample_rate = time_disp_sec.mean()

    if filter_signals:
        # Apply low-pass filter to all signals but current and backlash
        # According to Shanon theorem, we cannot get frequencies above
        # Fs/2 = 1/0.012 / 2 = 41Hz
        # To keep only relevant signal information, we apply Shannon theorem
        # twice, it means we keep frequencies below Fs/2/2 = 20 Hz.
        # Hence we apply a 20 Hz low-pass filter (Butterworth) order 5 to get
        # quick attenuation.
        for column in main_dataframe.columns:
            if (joint_name in column and "Current" not in column and
                    "Backlash" not in column):
                # Replace "NaN" with 0
                main_dataframe[column] = main_dataframe[column].replace(
                    to_replace=numpy.nan, value=0.0)
                # Apply filter
                main_dataframe[column] = pandas.Series(
                    butter_lowpass_filter(
                        data=numpy.array(main_dataframe[column]),
                        cutoff=20,
                        fs=1.0 / sample_rate,
                        order=5))

    # Compute sine frequency in Hz, based on command signal
    spectrum = numpy.fft.fft(main_dataframe[pos_act_col])
    frequencies = numpy.fft.fftfreq(len(spectrum), sample_rate)
    # Take main positive frequency
    kpis['Frequency'] = frequencies[spectrum[:len(spectrum) // 2].argmax()]

    # Compute position delay, based on phase diff (in s)
    # To obtain positive delay between command and sensor,
    # compute correlation(sensor, actuator) and search for maximum on the
    # second half of the correlation result
    xcorr = numpy.correlate(
        main_dataframe[pos_sen_col], main_dataframe[pos_act_col], mode='full')
    kpis["PhaseShiftTime"] = (
        xcorr[xcorr.size // 2:].argmax()) * sample_rate  # Convert in time

    # force the phase shift to be in [-pi:pi]
    kpis['PhaseShiftAngle'] = (
        2.0 * numpy.pi * kpis["PhaseShiftTime"] * kpis['Frequency'])

    # Compute RMS diff (in %)
    rms_act = get_rms(main_dataframe[pos_act_col])
    rms_sen = get_rms(main_dataframe[pos_sen_col])
    kpis["PositionRmsError"] = abs(rms_act - rms_sen) / rms_act * 100

    # Compute gain
    kpis["Gain"] = rms_sen / rms_act

    # Compute the offset
    mean_act = main_dataframe[pos_act_col].mean()
    mean_sen = main_dataframe[pos_sen_col].mean()
    kpis["Offset"] = mean_sen - mean_act

    # Compute mean current
    kpis['MeanCurrent'] = main_dataframe[joint_name +
                                         "ElectricCurrentSensorValue"].mean()

    # Compute max current
    kpis['MaximumCurrent'] = main_dataframe[joint_name +
                                            "ElectricCurrentSensorValue"].max()

    # Compute mean temperature
    kpis['MeanTemperature'] = main_dataframe[joint_name +
                                             "TemperatureValue"].mean()

    # Maximum amplitude
    kpis['MaximumAmplitude'] = (main_dataframe[pos_act_col].max() -
                                main_dataframe[pos_act_col].min()) / 2

    # --------------------------------------------------------------------------
    # OUTPUT FINALIZATION
    # --------------------------------------------------------------------------
    # Each result must be independant, so copies are added to avoid
    # any common object among different steps. Otherwise, if an upper
    # layer program modifies one step, it can impact the others.
    result = ProcessingResult(main_dataframe.copy(),
                              copy.deepcopy(metadata),
                              kpis)
    return result


def process_steps_signal(csv_path, metadata, filter_bad_steps=True):
    """Process a .csv file to get motor control KPIs on step.

    Args:
        csv_path (str): Path to the .csv file to process.
          It must contain Position, Current, Temperature, of a joint which is
            doing steps. **Time must be in seconds.**
        metadata (dict): metadata for this csv file. Must contain the following keys:
          * joint_name (str): the name of the joint
          * posture_name (str): the name of the initial posture of the robot
          * body_id (str): the body ID of the tested robot
        filter_bad_steps: (bool) set to True (default) to discard steps that
            have inconsistencies and should not be taken into account. To know
            more about this filter, read signal.control.process_steps_signal
            This arguments has an effect on steps only.

    Returns:
        (list) List of ProcessingResult objects. 
          Each object corresponds to a single step of the signal.
          ProcessingResult contains 3 attributes:
            * dataframe: a pandas dataframe containing the signal of this step.
            * metadata: a dictionary of metadata.
            * kpis: a dictionary of computed KPIs.
    """
    # --------------------------------------------------------------------------
    # VARIABLES INITIALIZATION
    # --------------------------------------------------------------------------
    results = []

    metadata['data_type'] = 'step'

    joint_name = metadata['system_name']

    # --------------------------------------------------------------------------
    # FILES INITIALIZATION
    # --------------------------------------------------------------------------
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("Cannot find file at {}".format(csv_path))

    # Load the CSV into the main dataframe
    main_dataframe = pandas.read_csv(filepath_or_buffer=csv_path)

    # --------------------------------------------------------------------------
    # DATA PROCESSING
    # --------------------------------------------------------------------------
    pos_act_col = joint_name + "PositionActuatorValue"
    pos_sen_col = joint_name + "PositionSensorValue"

    # First, process the whole signal without distinguishing each step
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    # If they are not there, compute extra-signals that are not retrieved from
    # the robot.
    extra_cols = ("TimeDispersion", joint_name + "CommandSpeed",
                  joint_name + "CommandAcceleration",
                  joint_name + "SensorSpeed", joint_name + "SensorAcceleration")
    if not all([c in main_dataframe.columns for c in extra_cols]):
        # Compute time dispersion
        main_dataframe["TimeDispersion"] = main_dataframe["Time"].diff(1)
        time_disp_sec = main_dataframe["TimeDispersion"]
        # Compute command speed
        main_dataframe[joint_name + "CommandSpeed"] = (
            main_dataframe[pos_act_col].diff(1) / time_disp_sec)
        # Compute command acceleration
        main_dataframe[joint_name + "CommandAcceleration"] = (
            main_dataframe[joint_name + "CommandSpeed"].diff(1) /
            main_dataframe["TimeDispersion"])
        # Compute sensor speed
        main_dataframe[
            joint_name +
            "SensorSpeed"] = main_dataframe[pos_sen_col].diff(1) / time_disp_sec
        # Compute sensor acceleration
        main_dataframe[joint_name + "SensorAcceleration"] = (
            main_dataframe[joint_name + "SensorSpeed"].diff(1) /
            main_dataframe["TimeDispersion"])
    else:
        time_disp_sec = main_dataframe["TimeDispersion"]

    # Steps are now analysed one by one. Therefore first thing to do is to split
    # the whole signal into pieces, each containing only one step.
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    # Detect each step occurrence in the command (when the command speed is max).
    # Extract sample numbers and time at which each step occurs:
    # Detect all the speed command peaks
    peaks, peak_prop = find_peaks(
                x=abs(main_dataframe[joint_name + "CommandSpeed"]),
                height=0.5,
                distance=20)  # 20*0.012s = 220ms or 20*0.020s = 400ms

    # Each cycle is defined by two sample indexes, one before and one after the
    # step.
    # For each cycle, a tuple is defined: (sample_init, sample_end)
    # each step starts 5 samples before the command speed is different from zero
    step_occ = []
    for i in range(0, len(peaks)):
        try:
            # pki:= occurrence of previous peaks
            if i == 0:
                pki = 0
            else:
                pki = peaks[i - 1]
            # pkf:= occurrence of current peaks
            pkf = peaks[i]
            # find the start of the step between the two peaks
            speed = main_dataframe[joint_name + "CommandSpeed"].iloc[pki:pkf]
            speedcumsum = speed.cumsum()
            gb = speedcumsum.groupby(speedcumsum)
            idmax = gb.size().idxmax()
            step_start = gb.groups[idmax][-1]

            step_occ.append(step_start)
        except:  # go to the next peak occurence if there is an error
            continue

    # Build tuples for every step cycles
    step_cycles_list = [(step_occ[i], step_occ[i + 1])
                        for i in range(0, len(step_occ) - 1)] # yapf: disable

    # For each step cycle, compute useful information and store them
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    for cycle_number, (ki, kf) in enumerate(step_cycles_list):
        loc_ki = main_dataframe.index[ki]
        loc_kf = main_dataframe.index[kf]
        
        step_cycle_pos_sen = main_dataframe[pos_sen_col].loc[loc_ki:loc_kf]
        
        # This dictionary will contain every KPI of this step
        step_kpis = {}

        # CycleTimes:= for each cycle, a tuple: (t_init, t_end)
        step_kpis["CycleTimes"] = (main_dataframe["Time"].iloc[ki],
                                   main_dataframe["Time"].iloc[kf])

        # Initial and final values for command
        step_kpis["InitialCommandValue"] = main_dataframe[pos_act_col].iloc[ki]
        step_kpis["FinalCommandValue"] = main_dataframe[pos_act_col].iloc[kf]

        # Step amplitude for command
        step_kpis["CommandStepAmplitude"] = step_kpis[
            "FinalCommandValue"] - step_kpis["InitialCommandValue"]

        # Step direction : increasing or decreasing angle
        step_kpis["StepDirection"] = "Increasing" if step_kpis[
            "CommandStepAmplitude"] > 0 else "Decreasing"
        
        # Maximum speed for this cycle
        step_kpis["MaximumSpeed"] = numpy.max(
            abs(main_dataframe[joint_name + "SensorSpeed"].iloc[ki:kf]))

        # Step command speed
        if step_kpis["StepDirection"] == "Increasing":
            ramp_stop = main_dataframe.loc[loc_ki:loc_kf][
                main_dataframe.loc[loc_ki:loc_kf, pos_act_col] >= step_kpis["FinalCommandValue"]
                ].index[0]
        elif step_kpis["StepDirection"] == "Decreasing":
            ramp_stop = main_dataframe.loc[loc_ki:loc_kf][
                main_dataframe.loc[loc_ki:loc_kf, pos_act_col] <= step_kpis["FinalCommandValue"]
                ].index[0]
        ramp_duration = (main_dataframe.loc[ramp_stop, 'Time'] - main_dataframe.loc[loc_ki, 'Time'])
        step_kpis["RampDuration"] = ramp_duration
        step_kpis["CommandSpeed"] = step_kpis["CommandStepAmplitude"] / ramp_duration

        # Maximum acceleration for this cycle
        step_kpis["MaximumAcceleration"] = numpy.max(
            abs(main_dataframe[joint_name + "SensorAcceleration"].iloc[ki:kf]))

        # Maximum current for this cycle
        step_kpis["MaximumCurrent"] = numpy.max(
            abs(main_dataframe[joint_name +
                               "ElectricCurrentSensorValue"].iloc[ki:kf]))

        # Temperature at the beginning of the cycle
        step_kpis["Temperature"] = main_dataframe[joint_name +
                                                  "TemperatureValue"].iloc[ki]

        # Settling time --------------------------------------------------------
        # To determine the settling time, we don't want to use an error band because
        # this band can vary a lot between joints and we don't have any absolute criteria
        # for this error. Therefore, we decided to define the settling time as the time from
        # which the sensor value stays in the "noise band". The half width of this noise band
        # is set to 5% (% of the sensor amplitude).
        # TODO: Assess if this is relevant, IT COULD BE BETTER TO MAKE AN ABSOLUTE NOISE BAND ON
        # NON-NORMALIZED SIGNAL. For instance, 0.015 rad to avoid the noise of the MRE.
        noise = 0.05  # 5% of normalized signal

        # **Explanation of the algorithm:**
        # Mean is computed for a part of the sensor signal.
        # If the maximum difference between this mean and the signal is lower or equal to the noise,
        # it means that this signal part is in the noise band.
        # If the maximum difference exceeds the noise band, a new part of the signal is selected
        # to compute a new fit and a new difference.
        # To select different parts of the signal, a dichotomy (only keeping right part) is done on
        # the full step signal.
        # Then, when the successful part is found, it is extend to the left until its maximum size
        # before the fit test fails.
        # This allows to find the exact portion of the signal that is static,
        # and thus the settling time.

        def _find_steady_state(signal, noise_ratio, step_init_value):
            """Recursive function which finds the steady state of a step signal.

            Args:
                signal: (Series) Signal to analyze.
                noise_ratio: (float) value of the noise band / 2,
                    in % of the steady state mean value.
                step_init_value: (float) the initial value of the signal
                    before the step. Used to compute the noise band.

            Returns:
                last_rec: (bool) True if this was the last recursion.
                index: (int) index of the dataframe from which the
                    steady state starts. 
                    numpy.nan if the steady has not been found.
                static_value: (float) mean value of the steady state.
                    numpy.nan if the steady has not been found.

            Warning:
                Returned index is an index label, to be used with loc[]
            """
            # Compute the mean of the signal
            mean = signal.mean()
            # Find maximum difference between this mean and the current part of the signal
            max_diff = max(abs(signal - mean))
            # Check if signal exceeds noise band
            noise_limit = noise_ratio * abs(mean - step_init_value)
            if max_diff > noise_limit:
                # Static part not found.
                # Recursively call this function with right part of the dichotomy
                middle = (
                    signal.index[0] + (signal.index[-1] - signal.index[0]) // 2)
                # Right part of the dichotomy
                subsig_start = middle
                subsig_stop = signal.index[-1]
                # If next signal has a length < 2 it means we did not find static part at all
                if subsig_stop - subsig_start < 2:
                    return True, numpy.nan, numpy.nan
                # Else, continue to search on the right part of the dichotomy
                last, index, static_value = _find_steady_state(
                    signal.loc[subsig_start:subsig_stop], noise_ratio, step_init_value)
                if last and index is not numpy.nan:
                    # Extend signal part to the left
                    # to be sure to catch the very beginning of the static part
                    noise_limit = noise_ratio * abs(static_value - step_init_value)
                    while (abs(signal.loc[index - 1] - static_value) <= noise_limit):
                        # Let's be careful not to exceed the left limit
                        if (index - 1) > signal.index[0]:
                            index -= 1
                        else:
                            break
                return False, index, static_value
            else:
                # Static part found ! And this is the last recursion.
                return True, signal.index[0], mean

        (last, loc_settling, static_value) = _find_steady_state(
            step_cycle_pos_sen, noise, step_cycle_pos_sen.iloc[0])
        if loc_settling is numpy.nan:
            step_kpis["SettlingTime"] = numpy.nan
        else:
            step_kpis["SettlingTime"] = (
                main_dataframe["Time"].loc[loc_settling] -
                main_dataframe["Time"].iloc[ki])
        # End of settling time -------------------------------------------------
        
        # Initial and final values for sensor
        step_kpis["InitialSensorValue"] = main_dataframe[pos_sen_col].iloc[ki]
        step_kpis["FinalSensorValue"] = numpy.mean(main_dataframe[pos_sen_col].loc[loc_settling:loc_kf])

        # Step amplitude for sensor
        step_kpis["SensorStepAmplitude"] = (step_kpis["FinalSensorValue"] - 
                                            step_kpis["InitialSensorValue"])
        
        # In order to compute 10% 90% rising times and overshoot we need
        # a normalized signal of the position sensor
        norm_pos_sen = ((step_cycle_pos_sen 
                         - step_kpis["InitialSensorValue"]) 
                        / step_kpis["SensorStepAmplitude"])
                
        # Rising time = t90 - t10 ----------------------------------------------
        # Measure the instant at which the output reaches 10% (t_10) of the amplitude of the sensor
        # …starting from the initial sensor value
        s_over = norm_pos_sen[norm_pos_sen > 0.1]
        if s_over.size > 0:
            loc_10 = s_over.index[0]
            step_kpis["RisingTime_0.1"] = (main_dataframe["Time"].loc[loc_10]
                                           - main_dataframe["Time"].iloc[ki])
        else:
            # output is never greater than 10%
            loc_10 = numpy.nan
            step_kpis["RisingTime_0.1"] = numpy.nan
        # Measure the instant at which the output attains 90% (t_90) of the amplitude of the sensor.
        # …starting from the initial sensor value
        s_over = norm_pos_sen[norm_pos_sen > 0.9]
        if s_over.size > 0:
            loc_90 = s_over.index[0]
            step_kpis["RisingTime_0.9"] = (main_dataframe["Time"].loc[loc_90]
                                           - main_dataframe["Time"].iloc[ki])
        else:
            # output is never greater than 90%
            step_kpis["RisingTime_0.9"] = numpy.nan
        # Compute the rise_time = t90 - t10. If one is "nan", then it will be "nan"
        if (step_kpis["RisingTime_0.1"] is numpy.nan 
            or step_kpis["RisingTime_0.9"] is numpy.nan):
            step_kpis["RisingTime_0.1_to_0.9"] = numpy.nan
            step_kpis["RisingSpeed"] = numpy.nan
        else:
            step_kpis["RisingTime_0.1_to_0.9"] = (
                step_kpis["RisingTime_0.9"] - step_kpis["RisingTime_0.1"])
            if step_kpis["RisingTime_0.1_to_0.9"] > 0:
                # Compute rising speed
                step_kpis["RisingSpeed"] = (
                    (0.8 * abs(step_kpis["SensorStepAmplitude"])) /
                    step_kpis["RisingTime_0.1_to_0.9"])
            else:
                step_kpis["RisingSpeed"] = numpy.nan
        # End of Compute the rising time ---------------------------------------
        
        # Overshoot ------------------------------------------------------------
        # check if there is an overshoot
        s_over = norm_pos_sen[norm_pos_sen > 1.0]
        if s_over.size > 0:
            # detect peaks:
            peaks, peak_prop = find_peaks(
                x=norm_pos_sen.loc[:loc_settling],
                height=1.0,
                distance=len(norm_pos_sen))
            if len(peaks) > 0:
                # Get overshoot time
                iloc_overshoot = ki + peaks[0]
                step_kpis["OvershootTime"] = (
                    main_dataframe['Time'].iloc[iloc_overshoot]
                    - main_dataframe['Time'].iloc[ki])
                # get overshoot magnitude in % and in rad
                step_kpis["Overshoot"] = (main_dataframe[pos_sen_col].iloc[iloc_overshoot] - 
                                          step_kpis["FinalSensorValue"])
                step_kpis["OvershootPercentage"] = abs(100.0 * (step_kpis["Overshoot"] /
                                                   step_kpis["SensorStepAmplitude"]))
            else:
                # peak was not correctly detected, this can induce error in KPI.
                step_kpis["OvershootTime"] = numpy.nan
                step_kpis["Overshoot"] = numpy.nan
                step_kpis["OvershootPercentage"] = numpy.nan
        else:
            step_kpis["OvershootTime"] = numpy.nan
            step_kpis["Overshoot"] = 0  # No overshoot: 0 rad
            step_kpis["OvershootPercentage"] = 0  # No overshoot: 0 %
        # End of Overshoot -----------------------------------------------------
        
        # Steady state current for this cycle ----------------------------------
        if loc_settling != numpy.nan:
            step_kpis["SteadyStateCurrent"] = numpy.mean(
                main_dataframe[joint_name + "ElectricCurrentSensorValue"].
                loc[loc_settling:loc_kf])
        else:
            step_kpis["SteadyStateCurrent"] = numpy.nan

        # Steady state error ---------------------------------------------------
        # difference between command and sensor once the system has settled
        # (from settling time to end of cycle)
        # Mean value is computed to avoid noise perturbation on final value
        if loc_settling != numpy.nan:
            step_kpis["SteadyStateError"] = numpy.mean(
                main_dataframe[pos_act_col].loc[loc_settling:loc_kf] -
                main_dataframe[pos_sen_col].loc[loc_settling:loc_kf])
            step_kpis["SteadyStateErrorPercentage"] = 100 * \
                (step_kpis["FinalSensorValue"] - step_kpis["FinalCommandValue"]) \
                / step_kpis["CommandStepAmplitude"]
        else:
            step_kpis["SteadyStateError"] = numpy.nan
        # End of steady state error --------------------------------------------

        # Energy of acceleration during the steady state -----------------------
        if loc_settling != numpy.nan:
            step_kpis["SteadyStateAccelerationEnergy"] = numpy.sum(
                numpy.square(main_dataframe[joint_name + "SensorAcceleration"].
                             loc[loc_settling:loc_kf]))
        else:
            step_kpis["SteadyStateAccelerationEnergy"] = numpy.nan
        # End of energy --------------------------------------------------------

        # Add the results of this step to the list of results
        # Only if this step is consistent and do not present some anomalies

        # Duration of the ramp (actual) must be consistent with the one written
        # in metadata (theorical). ±15%
        if (filter_bad_steps and
            not (metadata["transition_duration"] * 0.85
            <= ramp_duration * 1000  <=
            metadata["transition_duration"] * 1.15)):
            continue
        else:
            # Each result must be independant, so copies are added to avoid
            # any common object among different steps. Otherwise, if an upper
            # layer program modifies one step, it can impact the others.
            results.append(
                ProcessingResult(main_dataframe.iloc[ki:kf].copy(),
                                 copy.deepcopy(metadata),
                                 step_kpis))

        # End of processing for one step cycle -------------------------------------

    # End of processing all of the step cycles -------------------------------------

    # --------------------------------------------------------------------------
    # OUTPUT FINALIZATION
    # --------------------------------------------------------------------------
    return results
    # End of [action_process_upper_joint_step] --------------------------------


def get_normalized_position_timeseries(dataframe, command_offset,
                                       command_amplitude):
    """Get a noramlized position timeseries.
    
    Offset and amplitude could be computed from the dataframe, but as it is already
    done by process_steps_signal() it is not necessary to do it again here.
    
    Args:
        dataframe (pandas.DataFrame): the original dataframe
        command_offset (float): the initial offset of the command
        command_amplitude (float): the amplitude of the command

    Return:
        (pandas.DataFrame) A new dataframe with the following columns:
          Time, <joint>PositionActuatorValue, <joint>PositionSensorValue
          With an offset of 0, a step amplitude of 1, and a start time of 0.
    """
    # Get the columns of actuator and sensor position
    cols = [c for c in dataframe.columns if "Position" in c]

    # Get the time but remove the offset to make it start at 0
    time_series = dataframe['Time'] - dataframe['Time'].iloc[0]

    # Create the normalized dataframe with the Time only at first
    norm_df = pandas.DataFrame(time_series)

    # Add normalized position series to the normalized dataframe
    for col in cols:
        norm_df[col] = (dataframe[col] - command_offset) / command_amplitude

    # Set Time as index of this normalized dataframe
    norm_df.set_index('Time')

    return norm_df
