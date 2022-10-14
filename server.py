from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import os

import json
import socketserver

# Global constants
PORT = 8001

# Number of datapoints of past/future data to show on graph
PAST_DATAPOINTS = 12*24*3
FUTURE_DATAPOINTS = 12*24*3
# Indices where new data should be inserted into the array (overriding old
# data, as these will be more accurate / short term predicitons)
INSERTION_POINTS = [11, 12*6-1, 12*24-1, 12*48-1, 12*72-1];

# X values for plotting past datapoints
x = np.arange(0, PAST_DATAPOINTS, 1)
# X values for plotting future datapoints.
x_future = np.arange(PAST_DATAPOINTS, PAST_DATAPOINTS + FUTURE_DATAPOINTS, 1)

# Global variables for storing past/future data. Mean we don't have to keep
# declaring arrays over and over.
data_arr = np.zeros(PAST_DATAPOINTS)
future_arr = np.zeros(FUTURE_DATAPOINTS)

data_dict = {}
	# "warrandyte": {"past": data_arr, "future": future_arr},

# Files for loading/storing weather data.
PAST_DATA_FILE = "server_data/%s_past.np"
FUTURE_DATA_FILE = "server_data/%s_future.np"
LOCATIONS_FILE = "server_data/locations.json"

# Load list of locations from file.
locations = []
with open(LOCATIONS_FILE, "r") as f:
	locations = json.loads(("").join(f.readlines()))

# Location of graphs to update.
GRAPH_FILE = "webserver/graphs/%s.png"

# Format of datetime string.
DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

def unscale_data(data):
	"""
	Unscale data back to its original values for plotting.
	"""
	[temp, pres, hum, sec] = data
	temp = (temp * 60) - 10
	pres *= 100000
	hum *= 100
	sec *= 24 * 60 * 60

	return [temp, pres, hum, sec]

def unscale_predictions(predictions):
	"""
	Unscale predictions back to degrees celcius.
	"""
	return [(x * 60) - 10 for x in predictions]

def cycle_arr(arr, val):
	"""
	Move all datapoints in the array one to the left, inserting 'val' at the end.
	"""
	arr = np.roll(arr, -1)
	arr[-1:] = val;
	return arr

def cycle_predictions(arr, prediction):
	"""
	Move all datapoints in the array one to the left, inserting new predictions
	and the appropriate INSERTION_POINTS to override old predictions.
	"""
	arr = np.roll(arr, -1)
	for i in range(5):
		arr[INSERTION_POINTS[i]] = prediction[i]
	return arr

def load_data():
	"""
	Load stored data from file after a server reboot.
	"""

	for location in locations:
		if location not in data_dict:
			data_dict[location] = {}

		if os.path.exists(PAST_DATA_FILE % location):
			data_dict[location]['past'] = np.fromfile(PAST_DATA_FILE % location)
		else:
			data_dict[location]['past'] = np.zeros(PAST_DATAPOINTS)

		if os.path.exists(FUTURE_DATA_FILE % location):
			data_dict[location]['future'] = np.fromfile(FUTURE_DATA_FILE % location)
		else:
			data_dict[location]['future'] = np.zeros(FUTURE_DATAPOINTS)

def replot(location):
	"""
	Update the graph with the new data.
	"""

	# Update graph.
	print("Updated graph")
	plt.figure()
	plt.plot(x, data_dict[location]['past'])
	plt.plot(x_future, data_dict[location]['future'])
	plt.ylabel("Temperature (˚C)")
	timeStr = datetime.now().strftime("%H:%M")
	plt.xlabel("Updated %s" % timeStr)
	plt.title("Current Weather Data")
	plt.savefig(GRAPH_FILE % location)
	plt.close()

class TCPHandler(socketserver.StreamRequestHandler):
	"""
	Basic request handler for communicating with edge devices
	"""

	def handle(self):
		self.data = json.loads(self.rfile.readline().strip())
		print("Received request from '%s'" % self.client_address[0])

		# Extract data from request
		weather_data = unscale_data(self.data['weather_data'][0])
		predictions = unscale_predictions(self.data['prediction'][0])
		location = self.data['location']
		dateStr = self.data['date_str']

		if location not in locations:
			# Add new location
			locations.append(location)

			with open(LOCATIONS_FILE, "w") as f:
				f.write(json.dumps(locations))

			data_dict[location]['past'] = np.zeros(PAST_DATAPOINTS)
			data_dict[location]['future'] = np.zeros(FUTURE_DATAPOINTS)

		# Add new temperature data to the array
		data_dict[location]['past'] = cycle_arr(data_dict[location]['past'], weather_data[0])
		# Add new predictions to the future array
		data_dict[location]['future'] = cycle_predictions(data_dict[location]['future'], predictions)

		# Update data files
		data_dict[location]['past'].tofile(PAST_DATA_FILE % location)
		data_dict[location]['future'].tofile(FUTURE_DATA_FILE % location)

		# Update graph
		replot(location)


		# # Update data files.
		# with open(PAST_DATA_FILE % location, "w") as f:
			# data_di
			# # json.dumps(data_dict[location]['past'].to)
		# with open("web_weather_data.csv", "a") as f:
			# # Write weather data to file
			# line = (", ").join(str(x) for x in wd)
			# line = dateStr + ", " + line + "\n"
			# f.write(line)

		# # with open(FUTURE_DATA_FILE % loation, "w") as f:
		# with open(FUTURE_DATA_FILE, "a") as f:
			# # Write predictions to file
			# line = (", ").join(str(x) for x in od)
			# line = dateStr + ", " + line + "\n"
			# f.write(line)


def main():
	with socketserver.TCPServer(("", PORT), TCPHandler) as server:
		print("Hosting TCP Socket server on port '%d'" % PORT)
		load_data()
		print("Loaded data from file")
		server.serve_forever()

if __name__ == "__main__":
	main()
