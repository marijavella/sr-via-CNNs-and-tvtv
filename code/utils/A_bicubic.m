function [y] = A_bicubic(x, scaling_factor,M,N)

% This function computes A*x and returns the LR image y. In this case, the LR image is obtained 
% using the Matlab function imresize and the bicubic kernel.

x = reshape(x,M,N);
y = imresize(x,1/scaling_factor);
y= y(:);

end

