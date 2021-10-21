% ---------------------------------------------------------------------------
% Simple experiment, using only two images and SRCNN
% ---------------------------------------------------------------------------
clear
main('../images/SampleImages/SampleImages_SRCNN', ...
  '../images/SampleImages/SampleImages_HR', 2, 'RGB', 'ResultsTest.mat')

% --------------------------------------------------------------------------------------------
% Create a folder 'Test_Sets_Side' and download the images from the Google drive link provided
% Uncomment the following to replicate all the experiments in the paper
% --------------------------------------------------------------------------------------------
% main('../images/Test_Sets_Side/SRCNNSet5x2/', ...
% '../images/Test_Sets_GT/Set5/Set5x2', 2, 'Y', 'SRCNNSet5x2_Results.mat');

% main('../images/Test_Sets_Side/IRCNNSet5x4/', ...
% '../images/Test_Sets_GT/Set5/Set5x4', 4, 'RGB', 'IRCNNSet5x4_Results.mat');

% main('../images/Test_Sets_Side/and_so_on/', ...
% '../images/Test_Sets_GT/and_so_on', 4, 'Y or RGB', 'Results.mat');



