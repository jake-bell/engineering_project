% Test a model and save a plot of the results. Uncomment the lines for each
% exported model with the same name you want to test.

% Import test data from file.
TEST_DATA_FILE = "weather_data_out/test.csv";
global tableTest; tableTest = readtable(TEST_DATA_FILE);
% Import training data from file.
TRAINING_DATA_FILE = "weather_data_out/training.csv";
global tableTraining; tableTraining = readtable(TRAINING_DATA_FILE);

% Folder to store the output figures in. Any existing files with the same name will be overwritten.
global FIGURE_FOLDER; FIGURE_FOLDER = "matlab_figures/";

% Extract results from test data
global test_tempC; test_tempC = tableTest.tempC;
global training_tempC; training_tempC = tableTraining.tempC;

% Construct corresponding x values for plotting
global testlen; testlen = size(test_tempC);
global traininglen; traininglen = size(training_tempC);
global training_x; training_x = [1:traininglen];
global test_x; test_x = [1:testlen];
global new_test_x; new_test_x = test_x + max(training_x);

%testModel(fine_tree, "Fine Tree", "fine_tree.png");
testModel(wide_neural, "Wide Neural Network", "wide_neural.png");
%testModel(bilayered_neural, "Bilayered Neural Network", "bilayered_neural.png");
%testModel(trilayered_neural, "Trilayered Neural Network", "trilayered_neural.png");
%testModel(narrow_neural, "Narrow Neural Network", "narrow_neural.png");
%testModel(gaussian, "Gaussian SVM", "gaussian.png");
%testModel(medium_neural, "Medium Neural Network");

figure('Visible', 'on');

function testModel(model, modelTitle, modelFile)
	% Import used global variables
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

	% Set figure visibilty to false as we're saving the figure to file.
	fig = figure('Visible', 'off');
	% Offset all x values so that test data is plotted beside training data.
	new_test_x = test_x + max(training_x);
	% Plot results for comparison.
	plot(training_x, training_tempC, new_test_x, test_tempC, training_x, yfit_training, new_test_x, yfit);

	% Add plot details.
	legend("Training Data", "Test Data", "Training Prediction", "Test Prediction");
	title(modelTitle);
	ylabel("Temperature (°C)");
	% Disable x axis labels, as they are just used for plotting.
	set(gca,'XTick',[]);

	% Save figure to file.
	saveas(gcf, FIGURE_FOLDER + modelFile);
	close(fig);
end
