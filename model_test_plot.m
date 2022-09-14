% This is a matlab file.
% Assumes you have exported a trained model called
% 'trainedModel'

% Import test data from file.
TEST_DATA_FILE = "weather_data_out/test.csv";
global tableTest; tableTest = readtable(TEST_DATA_FILE);
TRAINING_DATA_FILE = "weather_data_out/training.csv";
global tableTraining; tableTraining = readtable(TRAINING_DATA_FILE);
global FIGURE_FOLDER; FIGURE_FOLDER = "figures/";

% Extract actual results from test data
global test_tempC; test_tempC = tableTest.tempC;
global training_tempC; training_tempC = tableTraining.tempC;

% Construct x values
global testlen; testlen = size(test_tempC);
global traininglen; traininglen = size(training_tempC);
global training_x; training_x = [1:traininglen];
global test_x; test_x = [1:testlen];
global new_test_x; new_test_x = test_x + max(training_x);

%figure;
%plot(tableTest.day * 86400 + tableTest.seconds, tableTest.tempC, tableTraining.day * 86400 + tableTraining.seconds, tableTraining.tempC);
%legend("test data", "training data");

testModel(fine_tree, "Fine Tree", "fine_tree.png");
testModel(wide_neural, "Wide Neural Network", "wide_neural.png");
testModel(bilayered_neural, "Bilayered Neural Network", "bilayered_neural.png");
testModel(trilayered_neural, "Trilayered Neural Network", "trilayered_neural.png");
testModel(narrow_neural, "Narrow Neural Network", "narrow_neural.png");
testModel(gaussian, "Gaussian SVM", "gaussian.png");
%testModel(medium_neural, "Medium Neural Network");
figure('Visible', 'on');

function testModel(model, modelTitle, modelFile)
	global tableTest;
	global tableTraining;
	global test_tempC;
	global training_tempC;
	global test_x;
	global training_x;
	global new_test_x;
	global FIGURE_FOLDER;

	% Use exported model to predict based on test data
	yfit = model.predictFcn(tableTest);

	% Re-predict based on training data so we can plot the results for comparison
	yfit_training = model.predictFcn(tableTraining);

	% Plot results for comparison.
	fig = figure('Visible', 'off');
	new_test_x = test_x + max(training_x);
	plot(training_x, training_tempC, new_test_x, test_tempC, training_x, yfit_training, new_test_x, yfit);
	legend("training", "test", "predictTrain", "predictTest");
	title(modelTitle);
	saveas(gcf, FIGURE_FOLDER + modelFile);
	close(fig);

end
