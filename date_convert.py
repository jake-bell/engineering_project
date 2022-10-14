from datetime import datetime
from os import path

# Outdated script, now merged into train_model.py

# Process/reformat data to play more nicely with matlab ML models and split
# into training and testing data.

# Get folder this script is in
SCRIPT_FOLDER = path.dirname(__file__)
# Folder for output data files
OUTPUT_FOLDER = "weather_data_out"
# Names/paths of output files
TEST_OUTPUT_FILE_NAME = "test.csv"
TRAINING_OUTPUT_FILE_NAME = "training.csv"
DATA_FILE_NAME = "weather_data.csv"
DATA_FILE = path.join(SCRIPT_FOLDER, DATA_FILE_NAME)
TEST_OUTPUT_FILE = path.join(SCRIPT_FOLDER, OUTPUT_FOLDER, TEST_OUTPUT_FILE_NAME)
TRAINING_OUTPUT_FILE = path.join(SCRIPT_FOLDER, OUTPUT_FOLDER, TRAINING_OUTPUT_FILE_NAME)
TEST_DATA_START = datetime(
	year=2022,
	month=9,
	day=5,
	#day=1,
	hour=15,
	minute=32
)

# Time when data first started being collected.
# Data will be updated to include fields for days since initial data, seconds since midnight of that day.
DATE_START = datetime(year=2022, month=8, day=25)
HEADER_LINE = "day, seconds, tempC, presPa, humRH\n"
DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

#
# Test data
#

line_a = "2022/08/25 15:31:10, 14.470000, 101545.578125, 74.611328"
line_b = "2022/09/01 15:30:01, 13.730000, 101128.226562, 0.54997070"
date_a = "2022/08/25 15:31:10"
date_b = "2022/09/01 15:30:01"

#
# Functions
#

def processLine(line):
	# Extract data from line
	[dateString, tempC, presPa, humRH] = [x.strip() for x in line.split(",")]
	day, seconds = -1, -1

	# Convert data to floats for processing.
	tempC, presPa, humRH = float(tempC), float(presPa), float(humRH)

	# Convert relative humidity to a number between 0 and 1 to better represent it as a percentage.
	# (rather than a percentage float between 0-100)
	if humRH >= 1:
		humRH /= 100

	# Convert string to datetime object based on DATETIME_FORMAT format specifier
	date_object = datetime.strptime(dateString, DATETIME_FORMAT)

	# Subtract initial date so all days are relative to start of data.
	datediff = date_object - DATE_START

	# Separate seconds since midnight from date for better accuracy?
	day, seconds = datediff.days, datediff.seconds

	# Construct new line
	result = '%d, %d, %lf, %lf, %lf\n' % (day, seconds, tempC, presPa, humRH)

	return result, date_object


def main():
	with open(DATA_FILE, "r") as data_file:
		with open(TRAINING_OUTPUT_FILE, "w") as training_file:
			with open(TEST_OUTPUT_FILE, "w") as test_file:
				# Consume the header line
				data_file.readline()
				# Write a header line to the output files.
				training_file.write(HEADER_LINE)
				test_file.write(HEADER_LINE)

				for line in data_file.readlines():
					new_line, date = processLine(line)
					if date > TEST_DATA_START:
						test_file.write(new_line)
					else:
						training_file.write(new_line)

main()
