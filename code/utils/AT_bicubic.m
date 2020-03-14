function [x] = AT_bicubic(y, scaling_factor,M,N)

% This function computes A'*y. Here, the input image is upscaled using the Matlab function 
% imresize and the bicubic kernel.

y = reshape(y,M,N);
x= imresize(y,scaling_factor);
x= (x(:));

end

