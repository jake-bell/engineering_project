total_data = readtable("new_weather_data.csv");
original_tempC = total_data.tempC(1:2880);
total_tempC = total_data.tempC;
total_pres_kPa = total_data.presPa / 1000;
total_humRH = total_data.humRH;
factor = 0.8;

figure('Visible', 'off')
original_x = 0:2879;
original_x = original_x / (12*24);
plot(original_x, original_tempC);
set(gcf, 'position', [0, 0, 1024*factor, 576*factor]);
legend("Temperature");
title("Original Weather Data");
ylim([0 25]);
xlabel("Time (days)");
ylabel("Temperature (˚C)");
saveas(gcf, "original_weather_data.png");
close(gcf);

len = size(total_tempC);
len = len(1) - 1;
total_x = 0:len;
total_x = total_x / (12*24);

figure('Visible', 'off');
plot(total_x, total_tempC);
set(gcf, 'position', [0, 0, 1024*factor, 576*factor]);
legend("Temperature");
title("Weather Data");
ylim([0 25]);
xlabel("Time (days)");
ylabel("Temperature (˚C)");
saveas(gcf, "total_weather_data.png");
close(gcf);

figure('Visible', 'off');
plot(total_x, total_pres_kPa);
set(gcf, 'position', [0, 0, 1024*factor, 576*factor]);
legend("Pressure (kPa)");
title("Weather Data");
xlabel("Time (days)");
ylabel("Pressure (kPa)");
saveas(gcf, "total_pressure.png");
close(gcf);

figure('Visible', 'off');
plot(total_x, total_humRH);
set(gcf, 'position', [0, 0, 1024*factor, 576*factor]);
legend("Relative Humidity");
title("Weather Data");
xlabel("Time (days)");
ylabel("Relative Humidity (%)");
saveas(gcf, "total_humidity.png");
close(gcf);