function [x] =AT_simplesubsampling(y, scaling_factor,M,N)

% This function is used to work out A'*y. A zero matrix with the same size of the HR image
% is first created. Afterwards, the samples of the LR image (y) are set to replace the zero entries.  
 
n = M*N;
ncols = N;

if isscalar(n) == 1
    n=n;
end

    if isscalar(n) == 0 && ~isempty(size(n,2)) == 1
      n = size(n ,1);
    end
    
M = length(y);
N = scaling_factor*M*scaling_factor;

indices = 1 : scaling_factor : N;
indices2 = reshape(indices,[],ncols);
newindices = 1 : scaling_factor : ncols;
indices3 = indices2(:,newindices);
indices = indices3;

if N >  length(n) || N  < length(n)
    N = n;
end

x = zeros(N, 1);

x(indices) = y;



