function [Out] = ReadImages(path)
% This function takes as input the path of folder where the images are
% located and outputs a struct contianing these images. 

pattern   =  {'*.jpg','*.png','*.bmp'};
D =path;

S = [dir(fullfile(D,pattern{1})), dir(fullfile(D,pattern{2})), dir(fullfile(D,pattern{3}))];

for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    I = imread(F);
    Out(k).data = I;
end

if isempty(S)==1
error('Error: Check that the image folder is not empty, the correct path of the image folder has been set.')
end

end

