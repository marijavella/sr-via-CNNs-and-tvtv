% ---------------------------------------------------------------------------
% Simple experiment, using only two images and SRCNN
% ---------------------------------------------------------------------------
clear
main('../images/SampleImages/SampleImages_SRCNN', ...
  '../images/SampleImages/SampleImages_HR', 2, 'ResultsTest.mat', 'A_FORWARD', 'A_box.m', 'A_TRANSPOSE', 'AT_box.m')

% --------------------------------------------------------------------------------------------
% Create a folder 'Test_Sets_Side' and download the images from the Google drive link provided
% Uncomment the following to replicate all the experiments in the paper
% --------------------------------------------------------------------------------------------
% main('../images/Test_Sets_Side/DRCNBSD100x2/', ...
% '../images/Test_Sets_GT/BSD100_GT/BSD100_x2_GT', 2, 'DRCNBSD100x2_Results.mat');
%
% main('../images/Test_Sets_Side/DRCNSet14x4/', ...
% '../images/Test_Sets_GT/Set14_GT/Set14_x4_GT', 4, 'DRCNSet14x4_Results.mat');
%
% main('../images/Test_Sets_Side/and_so_on/', ...
% '../images/Test_Sets_GT/and_so_on', 4, 'Results.mat');



