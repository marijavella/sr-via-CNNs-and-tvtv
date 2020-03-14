function [y] = A_box(x, scaling_factor, M, N)

% This function computes A*x. Here, the input LR image is upscaled using the Matlab function 
% imresize and the box-shaped kernel.

x = reshape(x, M, N);
y = imresize(x,1/scaling_factor, 'box');
y= y(:);

end


