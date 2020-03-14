function [x] = AT_box(y, scaling_factor,M,N)

% This function computes A'*y. Here, the input image is upscaled using the Matlab function 
% imresize and the box-shaped kernel.

y = reshape(y,M,N);
x= imresize(y,scaling_factor,'box');
x= (x(:));

end

