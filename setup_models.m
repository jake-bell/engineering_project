% Matlab file to automatically train and export models
% based on weather data.
TRAINING_DATA_FILE = "weather_data_out/training.csv";
training_data = removevars(readtable(TRAINING_DATA_FILE), {'humRH', 'presPa'});

TEST_DATA_FILE = "weather_data_out/test.csv";
test_data = removevars(readtable(TEST_DATA_FILE), {'humRH', 'presPa'});

%regressionLearner(training_data, 'tempC', 'CrossVal', 'on', 'Holdout', 0.1);
% Resubstitution validation seems to be the best.
regressionLearner(training_data, 'tempC', 'CrossVal', 'off');
%regressionLearner(training_data, 'tempC', 'CrossVal', 'on', 'KFolds', 3);


% Process:
% run setup_models
% Import. Train all quick to train, all neural.
% Export fine_tree, bilayered_neural, wide_neural
% run model_test_plot
% Figures will be saved as pngs under 'figures'
