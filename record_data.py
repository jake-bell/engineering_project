from datetime import datetime
from PiicoDev_BME280 import PiicoDev_BME280 as Sensor

# Note that this must be an absolute path to work with cron.
WEATHER_FILE = "/home/pi/code/weather_data/auto_capture.csv"

# Get a string representation of the current time.
def get_time():
	time_format_string = "%Y/%m/%d %H:%M:%S"
	now = datetime.now()
	time = now.strftime(time_format_string)
	return time

def main():
	# Get current time string
	current_time = get_time()

	# Get current sensor information
	sensor = Sensor()
	tempC, presPa, humRH = sensor.values()

	# Open file for appending
	with open(WEATHER_FILE, "a") as f:
		# Write date/time and sensor data to file.
		f.write("%s, %lf, %f, %f\n" % (current_time, tempC, presPa, humRH))

if __name__ == "__main__":
	main()
