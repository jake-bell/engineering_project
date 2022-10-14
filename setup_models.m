% Matlab file to automatically import data and setup training. Unfortunately,
% the actual training and model exporting must be done manually, but this will
% make the process easier.

TRAINING_DATA_FILE = "output_data.csv";
training_data = readtable(TRAINING_DATA_FILE);

%TEST_DATA_FILE = "weather_data_out/test.csv";
%test_data = removevars(readtable(TEST_DATA_FILE), {'humRH', 'presPa'});

% Test with different parameters for training by uncommenting only the applicable line.
% Resubstitution validation seems to be the best.
%regressionLearner(training_data, 'tempC', 'CrossVal', 'on', 'Holdout', 0.1);
regressionLearner(training_data, 'tempC', 'CrossVal', 'off');
%regressionLearner(training_data, 'tempC', 'CrossVal', 'on', 'KFolds', 3);


% Process:
% run setup_models
% Import. Train all quick to train, all neural.
% Export desired models with the correct names.
% run model_test_plot
% Figures will be saved as pngs under 'figures'
