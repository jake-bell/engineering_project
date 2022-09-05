% This is a matlab file.
% Assumes you have exported a trained model called
% 'trainedModel'
TEST_DATA_FILE = "weather_data_out/test.csv"
%date = [8; 8; 8; 8];
%time = [5; 10; 15; 20];
%T = table(date, time);
tableTest = readtable(TEST_DATA_FILE)
yfit = trainedModel.predictFcn(T);
