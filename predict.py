import numpy as np
import tflite_runtime.interpreter as tflite
import socket
from datetime import datetime
import json

# Should be a single line with the most recently recorded weather data stored in it for ease of use.
DATA_FILE = "/home/pi/code/weather_data/current_data.csv"
# HOST = "10.0.0.31"
HOST = "172.20.10.9"

def scale_data(temp, pres, hum, seconds):
	seconds = seconds / (24*60*60)
	temp = (temp + 10) / 60
	pres = pres / 100000
	hum = hum / 100
	return temp, pres, hum, seconds

def load_data():
	data = np.zeros((1, 4), dtype=np.float32)
	# Load and scale data.
	with open(DATA_FILE, "r") as f:
		# Read data
		line = f.readline()
		# Extract info
		[dateStr, tempC, presPa, humRH] = [x.strip() for x in line.split(",")]
		tempC, presPa, humRH = float(tempC), float(presPa), float(humRH)
		# Construct date
		date_object = datetime.strptime(dateStr, "%Y/%m/%d %H:%M:%S")
		seconds = date_object.second + date_object.minute * 60 + date_object.hour * 3600
		# Scale data as the model expects
		data[:] =  scale_data(tempC, presPa, humRH, seconds)

	return data, dateStr

def invoke_model(interpreter, input_data, input_details, output_details):
	interpreter.set_tensor(input_details['index'], input_data)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details['index'])
	return output_data

def main():
	interpreter = tflite.Interpreter(model_path='model.tflite')
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]
	input_shape = input_details['shape']
	data, dateStr = load_data()
	output_data = invoke_model(interpreter, data, input_details, output_details)

	print(data)
	print(output_data)

	# Convert data to json
	json_data = json.dumps({
		"date_str": dateStr,
		"weather_data": data.tolist(),
		"prediction": output_data.tolist(),
		"location": "warrandyte"
	})

	# print(json_data)

	# Send data to server
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		try:
			sock.connect((HOST, 8001))
			sock.sendall(bytes(json_data, "utf-8"))
		except ConnectionRefusedError as e:
			print("Server is offline, unable to send data")

main()
