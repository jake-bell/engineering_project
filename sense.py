from PiicoDev_BME280 import PiicoDev_BME280 as Sensor

# ------------------------------------- #
# Program for testing sensor connection #
# ------------------------------------- #

# Get the sensor
sensor = Sensor()

# Use a different hPa measurement of the mean sea level pressure to calculate the altitude.
# It's pretty innacurate if you're even slightly off, and this looks like it
# changes all the time, so it might be a bit difficult to get this accurately,
# and we don't need it anyway.
# MELBOURNE_SEA_LEVEL_PRESSURE_HPA = 1016
# altitude = sensor.altitude(pressure_sea_level=MELBOURNE_SEA_LEVEL_PRESSURE_HPA)

# Get sensor data
tempC, presPa, humRH = sensor.values()

# Print temp/pressure/relative humidity
print(("Temperature: %.1f\u00b0C, Pressure: %.2fkPa, Relative Humidity: %.1f%%") % (
	tempC, presPa / 1000, humRH
))

# Sample altitude a bunch of times and display the average.
# altitudes = []
# attempts = 20
# for i in range(attempts):
	# altitudes.append(sensor.altitude(pressure_sea_level=MELBOURNE_SEA_LEVEL_PRESSURE_HPA))
# amax, amin, aavg = max(altitudes), min(altitudes), sum(altitudes) / attempts
# print(("Altitudes (20 attempts), max:%.1fm, min:%.1fm, avg:%.1fm") % (amax, amin, aavg))

