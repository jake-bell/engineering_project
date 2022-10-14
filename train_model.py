from datetime import datetime, timedelta
from os import path

import numpy as np
from matplotlib import pyplot as plt

from keras import Input
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf

# TODO: Separate this into several parts, so that preprocessing, training the
# model, exporting the model, running the model, and plotting the results
# happen independently and can be run individually without taking a billion
# years to retrain the model each time when it's not necessary

#
# Global vars / Constants
#

# Set to true to plot errors on each output plot.
PLOT_ERRORS = True

# Set to false to just process data and exit, without spending ages training the model
TRAIN = True

# File to load training/test data from
DATA_FILE = "new_weather_data.csv"
OUTPUT_DATA_FILE = "output_data.csv"
OUTPUT_GRAPHS_FOLDER = "model_graphs/"

# Days of data to use for training
DATA_TRAINING_DAYS = 23 # 23 without 4/5/6 day extas
# Total days of data available. Will be used for testing. If predicting 3 days
# out, this should be at least 3 greater than 'DATA_TRAINING_DAYS'. If
# predicting 6 days out it should be at least 6 greater, etc.
TOTAL_DATA_DAYS = 26

# Format of datetime string for extracting date information
DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

# Date of first datapoint to include in training data
DATA_START = datetime(
	year = 2022,
	month=8,
	day=26
)

# Data of last datapoint to include in test and training data.
DATA_END = DATA_START + timedelta(days=TOTAL_DATA_DAYS)

# Date of last datapoint in file. Mainly just included to throw an error early
# if there's not enough data available.
FINAL_DATAPOINT = datetime(
	year=2022,
	month=9,
	day=22,
	hour=11,
	minute=25
)

# Number of total datapoints used for training.
DATAPOINTS = 12 * 24 * DATA_TRAINING_DAYS

# Size of an input datapoint.
INPUTS = 4
# Labels of inputs
INPUT_LABELS = ["tempC", "presPa", "humRH", "seconds"]

# Number of outputs the model will try to predict
# Make sure len of offsets is >= OUTPUTS
OUTPUTS = 5
# Array of offsets. i.e. how much the temperature data should be shifted for
# each output. e.g. '12' means the first output is a prediction 1 hour from
# now, as there are 12 datapoints per hour. 12*6 means that the second output
# is a predction 6 hours from now. Therefore each output is a copy of the input
# temperature of the same length as 'datapoints' but starting later in the
# array by the below amount. i.e. ouptut 1 is temp_input[12:12+datapoints]
# Prediction offsets, predict temp 1hr, 6hr, 1 day, 2 day, and 3 day from now
# This can be longer than outputs, but only the first 'OUTPUTS' entries will be used.
OFFSETS = [12 * 1, 12 * 6, 12 * 24, 12 * 48, 12 * 72, 12 * 96, 12 * 120, 12 * 144]
OFFSET_LABELS = ['1hr', '6hr', '24hr', '48hr', '72hr', '96hr', '120hr', '144hr'];

# Width of line when plotting. Smaller values may make it easier to see the
# difference between the actual and predicted values. I think the default is 1.
LINE_WIDTH = 0.5

# Resolution of the graph in dots per inch. Default is 100, so 300 will be
# triple the resolution.
DPI = 300

# Throw an error now if there's not enough data available so we don't go
# through most of training just to crash after 10 minutes.
if DATA_END > FINAL_DATAPOINT:
	raise Exception("Not enough data for training")

def scale_data(temp, pres, hum, seconds):
	"""
	Scale all data to be in the range [0,1], as this is what ML models need to
	work with. If this isn't done the model won't break, but the results will
	be pretty bad.
	"""

	# Ranges of data
	TEMP_RANGE = [-10, 50];
	PRES_RANGE = [0, 100000];
	HUM_RANGE = [0, 100];
	# DAY_RANGE = [0, 365];
	SECOND_RANGE = [0, 24*60*60];

	# day = (day - DAY_RANGE[0]) / DAY_RANGE[1]
	seconds = (seconds - SECOND_RANGE[0]) / (SECOND_RANGE[1] - SECOND_RANGE[0])
	temp = (temp - TEMP_RANGE[0]) / (TEMP_RANGE[1] - TEMP_RANGE[0])
	pres = (pres - PRES_RANGE[0]) / (PRES_RANGE[1] - PRES_RANGE[0])
	hum = (hum - HUM_RANGE[0]) / (HUM_RANGE[1] - HUM_RANGE[0])

	return temp, pres, hum, seconds

def extract_data():
	"""
	Extract all data from file into a numpy array and scale it to be within the
	range [0,1].

	Note that because we're using a numpy array it really needs to be declared
	beforehand for performance, which is why the start/end points are hardcoded
	and not determined from file. Make sure to change them if the input data
	changes.
	"""

	with open (DATA_FILE, "r") as data_file:
		# Strip header line from file. This is important for matlab, but will
		# just cause crashes here.
		data_file.readline()

		# Init data array
		# We load the entire dataset first then use only parts of it for input.
		# Therefore we need to calculate the total number of datapoints
		# (including test data) that we're loading in order to initialise the
		# array to the correct size.
		TOTAL_DATAPOINTS = 12 * 24 * TOTAL_DATA_DAYS
		data_arr = np.zeros((TOTAL_DATAPOINTS, INPUTS))

		# Index counter. Used to stop early once we've loaded enough data. This
		# is because it is prohibitively expensive to expand the array to fit
		# more data, so once it's full we just ignore the rest. If you want
		# more data, make sure to set the start/end points correctly above.
		i = 0

		# For each line of input
		for line in data_file.readlines():
			# Extract all input data as strings
			[dateStr, tempC, presPa, humRH] = [x.strip() for x in line.split(",")]

			# Get a date object based on the date string
			date_object = datetime.strptime(dateStr, DATETIME_FORMAT)

			# Skip the first few elements so we start on a new day (as set by
			# 'DATA_START'). Just cause. Looks nicer.
			if date_object > DATA_START and i < TOTAL_DATAPOINTS:
				# Convert time of day to seconds since midnight.
				day, seconds = date_object.day, date_object.second + date_object.minute * 60 + date_object.hour * 3600
				# Convert string data to floats and scale all data to within
				# [0,1] based on its expected maximum range.
				tempC, presPa, humRH, seconds = scale_data(float(tempC), float(presPa), float(humRH), seconds)
				# Append data to the input array
				data_arr[i][:] = tempC, presPa, humRH, seconds

				# Increment the counter.
				i = i + 1
			elif i > TOTAL_DATAPOINTS:
				# Break when the array is full.
				break

		# Once data is loaded from file, construct input/output arrays.
		# These are subsets of the entire data, 'datapoints' long.
		input_arr = data_arr[:DATAPOINTS]

		# Output data. Consists of future temperature points at various offsets
		# as described above. Basically we copy 'DATAPOINTS' number of
		# temperature values from 'data' to each column of the output array,
		# offset in time by the specified amounts. This means that given input
		# data at time 0 (0am), the outputs will be the temperature at time 12
		# (1am), 12*6 (6am), 12*24( 0am the following day ), etc. This allows
		# the neural network to predict future temperature values given only
		# the current data.
		output_arr = np.zeros((DATAPOINTS, OUTPUTS))

		# Get offset temperature data as current output.
		# For each row of output data
		for i in range(DATAPOINTS):
			# Copy the temperature data offset by the specified amounts into
			# the output. This is why the input data array MUST be smaller than
			# the total data array length by at least the longest amount you're
			# predicting into the future.
			for j in range(OUTPUTS):
				output_arr[i][j] = data_arr[i + OFFSETS[j]][0]

	# Return all three arrays
	return input_arr, output_arr, data_arr

def save_preprocessed_data(input_arr, output_arr):
	"""
	Once data has been imported, preprocessed, and scaled, save it to file so
	it can be easily be reused or input to matlab for training.
	"""

	# Open output file for writing.
	with open(OUTPUT_DATA_FILE, "w") as output_file:
		# Construct header line to label data
		HEADER_LINE = (", ").join(INPUT_LABELS[:INPUTS])

		# Output data will be as extra columns alongside input data, so add
		# labels for ouput data as well.
		for i in range(OUTPUTS):
			HEADER_LINE += ", future_temp_%d" % (i + 1)

		HEADER_LINE += "\n"

		# Write header line to label data columns
		output_file.write(HEADER_LINE)

		# Write lines of data to file.
		for i in range(DATAPOINTS):
			line = "%f" % input_arr[i][0]
			for j in range(1, INPUTS):
				line += ", %f" % input_arr[i][j]
			for j in range(OUTPUTS):
				line += ", %f" % output_arr[i][j]
			output_file.write(line + "\n")

def plot_results(predictions, data_arr, filename):
	"""
	Plot predictions vs input data and save figure to file with the specified
	name.
	"""

	# X values used only for plotting.
	xranges = [] # X values for given output
	# For plotting we need to convert the [DATAPOINTS][OUTPUTS] array to
	# [OUTPUTS][DATAPOINTS]
	results = [] # Extracted results of given output
	# Array for calculating errors and optionally plotting them to the same
	# graph.
	errors = [] # Error plot

	# Setup x ranges for data predictions based on their offsets.
	for i in range(OUTPUTS):
		xranges.append(np.arange(OFFSETS[i], OFFSETS[i] + len(predictions), 1))
		results.append(np.zeros(len(predictions)))
		errors.append(np.zeros(len(predictions)))

	# Temperature array for plotting expected results. Needs to be unscaled to
	# show actual temperature values in degrees celcius when plotted instead of
	# the scaled values used for training/predictions.
	temp_arr = np.zeros(len(data_arr))
	for i in range(len(data_arr)):
		temp_arr[i] = (data_arr[i][0] * 60) - 10

	# Extract results into separate arrays so they can overlay the actual data
	# at the correct point.
	# While these could all be plotted on the same graph, it gets a bit
	# cluttered, so they're each plotted onto their own graph for clarity.
	for i in range(len(predictions)):
		for j in range(OUTPUTS):
			# Again, the predicted values need to be unscaled so that they can
			# be plotted in degrees celcius.
			# This is done before calculating the error so that it is also
			# measured in degrees celcius rather than the less comprehensible
			# scaled values.
			results[j][i] = (predictions[i][j] * 60) - 10
			errors[j][i] = np.abs(results[j][i] - temp_arr[i + OFFSETS[j]])

	# Save a plot of temperature vs each output.
	for i in range(OUTPUTS):
		# Print the average absolute error for each prediction offset for
		# comparison and evaluation.
		print("Avg absolute error %d: %lf" % (
			i+1, np.average(errors[i])
		))
		print("Max absolute error:%d: %lf" % (
			i+1, np.max(errors[i])
		))

		# Construct plot. The syntax is basically the same as for MATLAB.
		# Get a new figure
		plt.figure()
		# Plot all recorded (actual) temperature values.
		plt.plot(np.arange(0, len(temp_arr), 1), temp_arr, lw = LINE_WIDTH)
		# Plot results with their correct offset so they overlay the actual
		# temperature values they were predicting. This allows easier
		# comparison.
		plt.plot(xranges[i], results[i], lw = LINE_WIDTH)

		# Additionally plot the errors if specified.
		if PLOT_ERRORS:
			plt.plot(xranges[i], errors[i], lw = LINE_WIDTH)
			# Add an average abs error plot as well for comparison
			mae = np.average(errors[i])
			plt.plot(xranges[i], np.linspace(mae, mae, len(xranges[i])), lw = LINE_WIDTH)

			# Add error information to legend if plotting it as well
			plt.legend(['Temperature data', OFFSET_LABELS[i], 'Absolute Error', 'Average Absolute Error'])
		else:
			# Exclude error information from legend if it's not being plotted
			plt.legend(['Temperature data', OFFSET_LABELS[i]])

		# Populate plot information.
		plt.title("Neural Network Predictions vs Reality")
		plt.xlabel("Time (days)")
		plt.ylabel("Temperature (°C)")
		plt.xticks(np.arange(0, 12*24*TOTAL_DATA_DAYS + 1, 12*24*2), np.arange(0, TOTAL_DATA_DAYS + 1, 2))

		# Save figure to file. Each is given a unique name based on the offset
		# number and the specified filename.
		# Default dpi is 100, so this is triple the resolution.
		plt.savefig('%s_offset_%d.png' % (OUTPUT_GRAPHS_FOLDER + filename, i + 1), dpi=DPI)

def train_model(input_arr, output_arr):
	"""
	Train a model with the given input and output data.
	"""

	# Model Parameters Results Comparison.
	# Batch size, Validation percent: MSE, data amount.
	# 36, 0.3: 0.001262, full data (23 days)
	# 48, 0.3: 0.00144, full data (23 days)
	# 36, 0.2: 0.00105, full data (23 days)
	# 36, 0.15 0.000922, 0.00085, full data (23 days)
	# Extra neuron (36, 0.15), 0.000692, 0.000781
	# 36, 0.1: 0.0007
	# 36, 0:0.0005
	# The above results show that the prediction works best with a smaller
	# amount of data used for validation. However, even with 20% of the data
	# reserved for validation, it still performs very well.

	# Network parameters
	BATCH_SIZE = 36
	#VALIDATION_PERCENT = 0.10 # best results when this is 0 lol
	EPOCHS = 1000
	HIDDEN_NEURONS = [56, 80, 56]

	# Construct model structure. Three hidden layers with the number of neurons
	# specified above. All use relu as their activation function. The output
	# layer, however, uses a linear activation function as it's predicting a
	# continuous value. The output layer needs to have the same number of
	# neurons as the desired number of outputs.
	model = Sequential([
		Dense(HIDDEN_NEURONS[0], input_dim=INPUTS, activation='relu'),
		Dense(HIDDEN_NEURONS[1], activation='relu'),
		Dense(HIDDEN_NEURONS[2], activation='relu'),
		# Output neurons
		Dense(OUTPUTS, activation='linear')
	])

	# Compile and fit (train) the model on the provided data.
	model.compile(
		# Loss function to use
		loss='mse',
		# Optimiser funciton. Basically uses the loss function to improve
		# network values at each iteration. After testing, I found that this
		# one worked best.
		optimizer='adam',
		# Metrics for evaluating performance. 'mse' is mean squared error, 'mae' is mean absolute error
		metrics=['mse', 'mae']
	)

	# Train model on provided data.
	fit_results = model.fit(
		input_arr,
		output_arr,
		epochs = EPOCHS,
		batch_size = BATCH_SIZE,
		# validation_split = VALIDATION_PERCENT,
		verbose = 1
	)

	# Return the trained model
	return model

def export_model(model):
	"""
	Convert the trained model to TFlite format and write it to file.
	"""
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()

	with open('model.tflite', 'wb') as f:
		f.write(tflite_model)

def main():
	# Get input/output data from file.
	input_arr, output_arr, data_arr = extract_data()

	# Save processed data for further use.
	save_preprocessed_data(input_arr, output_arr)

	# Uncomment to use less data for training for comparison. We can still
	# plot/predict based on all of the available data.
	# Also still evaluates the model's performance based on the entire set of
	# data.
	# training_input = input_arr[:12*24*12]
	# training_output = output_arr[:12*24*12]
	training_input = input_arr
	training_output = output_arr

	# If we want to train the model, do so. Otherwise we're just running to
	# preprocess data and save it to file.
	if TRAIN:
		# Train model
		model = train_model(training_input, training_output)
		print("Finished training, making predictions.")

		# Make predictions based on the entire input array.
		train_results = model.predict(input_arr)
		# Print the accuracy (mse) of the model based on all the available data.
		print("Accuracy: ", model.evaluate(input_arr, output_arr, verbose=1))

		# Plot prediction results
		plot_results(train_results, data_arr, "training")

		# Export the trained model to file.
		export_model(model)

if __name__ == "__main__":
	main()

# Results: (3 days out) mse: mae:
#Accuracy:  [0.0003923093026969582, 0.0003923093026969582, 0.013911795802414417]
#Avg absolute error 1: 0.458884
#Avg absolute error 2: 0.808793
#Avg absolute error 3: 0.874664
#Avg absolute error 4: 0.972308
#Avg absolute error 5: 1.05889

# Results: (6 days out) mse: mae:
#Accuracy:  [0.0004640071711037308, 0.0004640071711037308, 0.015445717610418797]
#Avg absolute error 1: 0.442377
#Avg absolute error 2: 0.913636
#Avg absolute error 3: 1.012037
#Avg absolute error 4: 1.061623
#Avg absolute error 5: 1.063892
#Avg absolute error 6: 1.037697
#Avg absolute error 7: 0.936010
#Avg absolute error 8: 0.946673


# Results (3 days out) limited data (12 days out of 23-26)
#Accuracy:  [0.001467150286771357, 0.001467150286771357, 0.02419644221663475]
#Avg absolute error 1: 0.586192
#Avg absolute error 2: 1.319792
#Avg absolute error 3: 1.611480
#Avg absolute error 4: 1.847405
#Avg absolute error 5: 1.894065

# Results (6 days out) limited data (12 days instead of 20-26)
#Accuracy:  [0.0015409882180392742, 0.0015409882180392742, 0.0253153033554554]
#Avg absolute error 1: 0.555594
#Avg absolute error 2: 1.510517
#Avg absolute error 3: 1.613438
#Avg absolute error 4: 2.021968
#Avg absolute error 5: 1.773913
#Avg absolute error 6: 1.569724
#Avg absolute error 7: 1.418726
#Avg absolute error 8: 1.687469


# Matlab RMSE
# Wide 0.014279
# Trilayer 0.021687
