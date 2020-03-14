function [y] = A_simplesubsampling(x, scaling_factor,M,N)

% This function is used to works out A*x and returns the LR image b. In this case, the LR image 
% is obtained by sampling each k rows and columns where k is equal to the scaling factor. 

ncols = N;
N = length(x);

indices = 1 : scaling_factor : N;

if length(indices) > floor(N/scaling_factor)
    indices = indices(1:end-1);
end

  if length(indices) < floor(N/scaling_factor) 
     diff = (N/scaling_factor) - length(indices);
     indices(end+1) = indices(end) + scaling_factor;
  end
  
indices2 = reshape(indices,[],ncols);
newindices = 1 : scaling_factor : ncols;
indices3 = indices2(:,newindices);
y = x(indices3);
y = y(:);

end
