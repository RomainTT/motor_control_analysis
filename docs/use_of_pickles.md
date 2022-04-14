# Use of .pickle files

## Content

A `.pickle` contains an object with all the post-processed data of a single operating point. 
Therefore, one `.pickle` is always associated to a synthesis sheet.

In the next paragraphs, `data` is the name of the object in which the `.pickle` file is loaded.

Here is its content **for a sine**:

```python
data = {
	# type of data
	"data_type": "sine",
	# information about the operating point
	"operating_point": {
		"current_range": (<<class 'numpy.float64'>>, <<class 'numpy.float64'>>),
	 	"load_direction": <<class 'int'>>,
		"system_name": <<class 'str'>>,
		"temperature_range": (<<class 'numpy.float64'>>, <<class 'numpy.float64'>>)
	},
	# data of the chart "maximum current distribution"
	"maximum_current_dist": {
		"mean": <<class 'numpy.float64'>>,
	 	"median": <<class 'numpy.float64'>>,
	  	"q1": <<class 'numpy.float64'>>,
	   	"q3": <<class 'numpy.float64'>>,
	    "wmax": <<class 'numpy.float64'>>,
		"wmin": <<class 'numpy.float64'>>
	},
	# List of post-processed steps, with their raw data and KPIS
	"processing_results": [ProcessingResult]
}
```

Here is its content **for a step**:

```python
data = {
	# type of data
	"data_type": "step",
	# information about the operating point
	"operating_point": {
		"current_range": (<<class 'numpy.float64'>>, <<class 'numpy.float64'>>),
	 	"load_direction": <<class 'int'>>,
		"system_name": <<class 'str'>>,
		"temperature_range": (<<class 'numpy.float64'>>, <<class 'numpy.float64'>>)
	},
	# data of the chart "overshoot in function of speed"
	"overshoot_on_speed": {
		"disp_inner": <<class 'float'>>,
		"disp_outer": <<class 'float'>>,
		"fit_func_name": <<class 'str'>>,  # "first_order" or "inverse"
		"fit_func_param": <List[float]>,
		"fit_func_str": <<class 'str'>>,  # string representation of the function
		"nrmsd": <<class 'numpy.float64'>>  # normalized root mean square deviation
	},
	# data of the chart "overshoot time in function of speed"
	"overshoot_time_on_speed": {
		"disp_inner": <<class 'float'>>,
		"disp_outer": <<class 'float'>>,
		"fit_func_name": <<class 'str'>>,  # "first_order" or "inverse"
		"fit_func_param": <List[float]>,
		"fit_func_str": <<class 'str'>>,  # string representation of the function
		"nrmsd": <<class 'numpy.float64'>>  # normalized root mean square deviation
	},
	# data of the chart "rising time 10% in function of speed"
	"rising_time_10_on_speed": {
		"disp_inner": <<class 'float'>>,
		"disp_outer": <<class 'float'>>,
		"fit_func_name": <<class 'str'>>,  # "first_order" or "inverse"
		"fit_func_param": <List[float]>,
		"fit_func_str": <<class 'str'>>,  # string representation of the function
		"nrmsd": <<class 'numpy.float64'>>  # normalized root mean square deviation
	},
	# data of the chart "rising time 90% in function of speed"
	"rising_time_90_on_speed": {
		"disp_inner": <<class 'float'>>,
		"disp_outer": <<class 'float'>>,
		"fit_func_name": <<class 'str'>>,  # "first_order" or "inverse"
		"fit_func_param": <List[float]>,
		"fit_func_str": <<class 'str'>>,  # string representation of the function
		"nrmsd": <<class 'numpy.float64'>>  # normalized root mean square deviation
	},
	# data of the chart "settling time in function of speed"
	"settling_time_on_speed": {
		"disp_inner": <<class 'float'>>,
		"disp_outer": <<class 'float'>>,
		"fit_func_name": <<class 'str'>>,  # "first_order" or "inverse"
		"fit_func_param": <List[float]>,
		"fit_func_str": <<class 'str'>>,  # string representation of the function
		"nrmsd": <<class 'numpy.float64'>>  # normalized root mean square deviation
	},
	# data of the chart "maximum current distribution"
	"maximum_current_dist": {
		"mean": <<class 'numpy.float64'>>,
	 	"median": <<class 'numpy.float64'>>,
	  	"q1": <<class 'numpy.float64'>>,
	   	"q3": <<class 'numpy.float64'>>,
	    "wmax": <<class 'numpy.float64'>>,
		"wmin": <<class 'numpy.float64'>>
	},
	# data of the chart "steady state current distribution"
	"steady_state_current_dist": {
		"mean": <<class 'numpy.float64'>>,
	 	"median": <<class 'numpy.float64'>>,
	  	"q1": <<class 'numpy.float64'>>,
	   	"q3": <<class 'numpy.float64'>>,
	    "wmax": <<class 'numpy.float64'>>,
		"wmin": <<class 'numpy.float64'>>
	},
	# data of the chart "steady state current distribution"
	"steady_state_error_dist": {
		"mean": <<class 'numpy.float64'>>,
	 	"median": <<class 'numpy.float64'>>,
	  	"q1": <<class 'numpy.float64'>>,
	   	"q3": <<class 'numpy.float64'>>,
	    "wmax": <<class 'numpy.float64'>>,
		"wmin": <<class 'numpy.float64'>>
	},
	# data of the chart "overshoot distribution"
	"overshoot_dist": {
		"mean": <<class 'numpy.float64'>>,
	 	"median": <<class 'numpy.float64'>>,
	  	"q1": <<class 'numpy.float64'>>,
	   	"q3": <<class 'numpy.float64'>>,
	    "wmax": <<class 'numpy.float64'>>,
		"wmin": <<class 'numpy.float64'>>
	},
	# List of post-processed steps, with their raw data and KPIS
	"processing_results": [ProcessingResult]
}
```

Here is the content of a `ProcessingResult` object **for a sine**:

```python
processing_result.metadata = {
	"start_position": <<class 'str'>>,
	"robot_head_id": <<class 'str'>>,
	"robot_body_id": <<class 'str'>>,
	"system_name": <<class 'str'>>,
	"maximum_speed": <<class 'float'>>,  # deg/s
	"amplitude": <<class 'float'>>,  # deg
	"command_time_step": <<class 'float'>>,  # ms
	"data_type": "sine",
	"temperature_limit": <<class 'int'>>,
	"test_duration": <<class 'int'>>,
	"date": <<class 'str'>>
}

processing_result.kpis = {
	"PhaseShiftAngle": <<class 'numpy.float64'>>,
	"MaximumCurrent": <<class 'numpy.float64'>>,
	"MaximumAmplitude": <<class 'numpy.float64'>>,
	"MeanCurrent": <<class 'numpy.float64'>>,
	"Frequency": <<class 'numpy.float64'>>,
	"MeanTemperature": <<class 'numpy.float64'>>,
	"Offset": <<class 'numpy.float64'>>,
	"Gain": <<class 'numpy.float64'>>,
	"PositionRmsError": <<class 'numpy.float64'>>,
	"PhaseShiftTime": <<class 'numpy.float64'>>
}

# A pandas Dataframe containing the raw signal
processing_result.dataframe = <pandas.Dataframe>
```

Here is the content of a `ProcessingResult` object **for a step**:

```python
processing_result.metadata = {
	"start_position": <<class 'str'>>,
	"robot_head_id": <<class 'str'>>,
	"robot_body_id": <<class 'str'>>,
	"system_name": <<class 'str'>>,
	"maximum_speed": <<class 'float'>>,  # deg/s
	"amplitude": <<class 'float'>>,  # deg
	"data_type": "step",
	"test_duration": <<class 'int'>>,
	"transition_duration": <<class 'float'>>,
	"load_direction": <<class 'int'>>,
	"date": <<class 'str'>>
}

processing_result.kpis = {
	"RisingSpeed": <<class 'numpy.float64'>>
	"SteadyStateError": <<class 'numpy.float64'>>
	"MaximumCurrent": <<class 'numpy.float64'>>
	"RisingTime_0.1_to_0.9": <<class 'numpy.float64'>>
	"SteadyStateErrorPercentage": <<class 'numpy.float64'>>
	"MaximumAcceleration": <<class 'numpy.float64'>>
	"RisingTime_0.1": <<class 'numpy.float64'>>
	"SensorStepAmplitude": <<class 'numpy.float64'>>
	"MaximumSpeed": <<class 'numpy.float64'>>
	"RampDuration": <<class 'numpy.float64'>>
	"SteadyStateAccelerationEnergy": <<class 'numpy.float64'>>
	"CommandStepAmplitude": <<class 'numpy.float64'>>
	"OvershootPercentage": <<class 'float'>>
	"Temperature": <<class 'numpy.float64'>>
	"Overshoot": <<class 'float'>>
	"SteadyStateCurrent": <<class 'numpy.float64'>>
	"CycleTimes": <<class 'tuple'>>
	"InitialCommandValue": <<class 'numpy.float64'>>
	"StepDirection": <<class 'str'>>
	"RisingTime_0.9": <<class 'numpy.float64'>>
	"OvershootTime": <<class 'float'>>
	"FinalCommandValue": <<class 'numpy.float64'>>
	"FinalSensorValue": <<class 'numpy.float64'>>
	"CommandSpeed": <<class 'numpy.float64'>>
	"SettlingTime": <<class 'numpy.float64'>>
	"InitialSensorValue": <<class 'numpy.float64'>>
}

# A pandas Dataframe containing the raw signal
processing_result.dataframe = <pandas.Dataframe>
```

## Usage examples

If there is a need to separate data of an operating points into two groups with different 
static errors…

```python
from utils.data import load_op_data

data = load_op_data("build/pepper/HipPitch/synthesis_step_HipPitch_2.0-3.0_30.0-60.0_-1.pickle")

g1 = []
g2 = []

for step in data["processing_results"]: 
     coef = 1 if step.kpis["StepDirection"] == "Increasing" else -1 
     err = step.kpis["SteadyStateError"] * coef  
     if err > 0: 
         g1.append(step) 
     else: 
         g2.append(step) 
```

