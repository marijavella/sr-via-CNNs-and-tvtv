function [x_opt, k] = TVTV_Solver_GPU(M, N, b, w_im, beta, arg1, arg2,  scaling_factor)

% This is the equivalent GPU version of TVTV_Solver.m
% Solves
%                  minimize    TV(x) + beta*TV(x - w_im)
%                     x
%                  subject to  b = A*x
%
% where TV(x) is the 2D total variation of a vectorized version x of
% an M x N matrix X (i.e., x = vec(X)), b : m x 1 is a vector of
% measurements (b = A*x), beta > 0, and w_im is a vectorized image
% similar to the image we want to reconstruct. We use ADMM to solve
% the above problem, as explained in the attached documentation.
%
% Access to A is given implicitly through the function handler for the operations 
% A*x and A'*y (in arg1 and arg2, respectively). In this version of the code, A 
% represent the downscaling operation through bicubic interpolation. This can be 
% changed to be in line how b was obtained. We also provide the code for box 
% averaging and simple submsapling in utils folder. For box averaging arg1 - A_box.m 
% and arg2 - AT_box.m. For simple subsampling set arg1 - A_simplesubsampling.m and 
% arg2 - AT_simplesubsampling.m.  
%
% arg1 is a function handler to A*x
% arg2: a function handler to A'*y
%               
% Inputs:
%   - M:    number of rows of the original image
%   - N:    number of columns of the original image (n = M*N)
%   - b:    vector of measurements (of size m)
%   - w_im: n x 1 vector representing an image W_im: w_im = vec(W_im)
%   - beta: positive number
%   - arg1: a function handler
%   - arg2: a function handler
%   - scaling_factor: scaling factor
%
% Outputs:
%   - x_opt: solution of the above optimization problem
%   - k: number of ADMM iterations
%
% This code was designed and implemented by M. Vella to perform experiments 
% described in
%
% [1] M. Vella, J. F. C. Mota
%     Robust Single-Image Super-Resolution via CNNs and TV-TV Minimization
%     preprint: Insert URL
%     2020
%
% Contributors:
%     Marija Vella
%     Joao F. C. Mota
%
% =========================================================================
% TVTV_Solver: minimizing TV+TV with linear constraints
% Copyright (C) 2020  Marija Vella
% 
% 
% =========================================================================
%
% =========================================================================
% Any feedback, comments, bugs and questions are welcome and can be sent to mv37@hw.ac.uk
% =========================================================================

% =========================================================================
% Check inputs

n = M*N;
m = length(b);

if length(w_im) ~= n  || beta < 0
    error('Input dimensions of TvplusTv do not agree. Type ''help TvplusTv'' for more information.');
end
    A  = @(x,scaling_factor,M,N) arg1(x, scaling_factor, M,N);
    AT = @(x,scaling_factor,M,N) arg2(x, scaling_factor, M,N);

% =========================================================================

% =========================================================================
% Parameters
MAX_ITER = 1700;
rho      = 0.5;
tau_rho  = 10;
mu_rho   = 2;

eps_prim = 1e-3;
eps_dual = 1e-3;
% =========================================================================

% =========================================================================
% Initializations and precomputations

% Vectors c_h and c_v defining the circulant matrices
c_h  = zeros(n, 1);
c_h(1)  = -1;
c_h(n-M+1)  = 1;
c_h = gpuArray(c_h); % transforming to gpu array

c_v  = zeros(n, 1);
c_v(1) = -1;
c_v(n) = 1;
c_v = gpuArray(c_v);

Fc_h = fft(c_h);
Fc_v = fft(c_v);

% Squaring the diagonal matrices Fc_h and Fc_v

Fc_v_diag = real(Fc_v);
Fc_v_diag_square = Fc_v_diag .^2; % vector containing diagonal entries squared

Fc_h_diag = real(Fc_h);
Fc_h_diag_square = Fc_h_diag .^2; % vector containing diagonal entries squared

h = 1./(Fc_v_diag_square+Fc_h_diag_square +1);
h = gpuArray(h);

% -------------------------------------
% Multiplication by D and D'

D_h  = @(z) real(ifft(Fc_h .* fft(z)));
DT_h = @(z) real( fft(Fc_h .* ifft(z)));

D_v  = @(z) real(ifft(Fc_v .* fft(z)));
DT_v = @(z) real( fft(Fc_v .* ifft(z)));

D    = @(z) [D_v(z) ; D_h(z)];
DT   = @(z) DT_v(z(1:n)) + DT_h(z(n+1:2*n));

% See definition of multiply_B at the end of file
B = @(z,A,AT, scaling_factor,M,N) multiply_B(z,A,AT, scaling_factor,M,N);

% ----------------------------------------------------------------------------

% Transform the side information w_im into the domain of derivatives
w = D(gpuArray(w_im));

% Initialization of ADMM variables
x = w_im; % x = (nx1)
u = w; % u = 2nx1
v = x; % v = (nx1)

lambda = zeros(2*n,1);
mu = zeros(n,1);
r_prim = zeros(3*n,1);
s_dual = zeros(3*n,1);
v_aux = zeros(n, 1);

% For conjugate gradient
c_aux = zeros(m, 1);
g = zeros(n,1);
p = zeros(n,1);

% =========================================================================

for k = 1 : MAX_ITER
    
    % ********************************************************************

    % Minimization in u

    s = lambda - (rho*D(gpuArray(v))); 
    
    rhow = rho*w;
    
    w_pos = (w >= 0);
    
    % ------------------------------------------------------------
    % Components for which w_i >= 0
    
    case1 = logical((w_pos) .* (s < -rhow -beta -1));
    u(case1) = ( -beta - 1 - s(case1) )/rho;
    
    case2 = logical(w_pos .* (-rhow - beta - 1 <= s) .* (s <= -rho*w + beta - 1));
    u(case2) =  w(case2);
    
    case3 = logical(w_pos .* (-rhow + beta - 1 < s) .* (s < beta - 1));
    u(case3) = (beta - 1 - s(case3))/rho;
    
    case4 = logical(w_pos .* (beta - 1 <= s) .* (s <= beta + 1));
    u(case4) = 0;

    case5 = logical(w_pos .* (s > beta + 1));
    u(case5) = (beta + 1 - s(case5))/rho;
    
    % ------------------------------------------------------------
    
    % ------------------------------------------------------------
    % Components for which w_i < 0
    
    case1r = logical(~w_pos .* (s < -beta -1));
    u(case1r) = (-beta -1 - s(case1r))/rho;
    
    case2r = logical(~w_pos .* (-beta - 1 <= s) .* (s <= -beta + 1));
    u(case2r) = 0;
    
    case3r = logical(~w_pos .* (-beta + 1 < s) .* (s < -rho*w - beta + 1));
    u(case3r) = (-beta + 1 - s(case3r))/rho;
    
    case4r = logical(~w_pos .* (-rho*w - beta + 1 <= s) .* (s <= -rho*w + beta + 1));
    u(case4r) = w(case4r);
    
    case5r = logical(~w_pos .* (s > -rho*w + beta + 1));
    u(case5r) = (beta + 1 - s(case5r))/rho;
    
    u_bar = u;
    % *********************************************************************
    % Minimization in x
    
        p = gather( (1/rho)*( (rho*(v)) - mu));
        Apb = A(p,scaling_factor,M,N) - b;
        
        if strcmp(char(arg1), 'A_simplesubsampling' ) ==  1
            c_aux2 = AT(Apb,scaling_factor,M,N);
        end
        
        if  strcmp(char(arg1), 'A_bicubic')
            c_aux = conjgrad(B, Apb,c_aux);
            c_aux2 = AT(c_aux, scaling_factor, M/scaling_factor, N/scaling_factor);
        end
        
        if strcmp(char(arg1), 'A_box' ) ==  1
            c_aux2 = (1/(scaling_factor^2))*AT(Apb,scaling_factor,M,N);
        end
       
        x = (p-(c_aux2));
        x_bar = x;
        
    % *********************************************************************
    % Minimization in v

    if k ==1
       v_bar = v;
    else
       v_bar = v_aux;
    end

    v_bar_prev = v_bar;
    g = DT(gpuArray(u_bar +(1/rho)*lambda)) + (1/rho)*mu + x_bar;
    v_aux = ifft(h.*fft(g));

    v_bar = v_aux;
    
    % *********************************************************************
    % Update dual variable
    
    lambda = lambda;
    mu = mu;
    
    r_prim = [u_bar; x_bar] - [D(gpuArray(v_bar)); v_bar] ;    % primal residual
    lambda = lambda + rho*r_prim(1:2*n);
    mu = mu + rho*r_prim(2*n+1:3*n);
    
    s_dual = -rho*[D(gpuArray(v_bar + v_bar_prev)); (v_bar + v_bar_prev) ]; % dual residual

    % *********************************************************************
    
    % *********************************************************************
    % rho adjustment
    
    r_prim_norm = norm(gpuArray(r_prim));
    primal_residual(k) = gather(r_prim_norm);
    s_dual_norm = norm(gpuArray(s_dual));

    if r_prim_norm > tau_rho*s_dual_norm
        rho = mu_rho*rho;
    elseif s_dual_norm > tau_rho*r_prim_norm
        rho = rho/mu_rho;
    end
    % *********************************************************************

      
    if r_prim_norm < eps_prim && s_dual_norm < eps_dual
        break;
    end
end

x_opt = real(x_bar);

if k >= MAX_ITER
    fprintf('Warning: Maximum number of iterations reached. Primal residual = %f, Dual residual = %f\n', ...
        r_prim_norm, s_dual_norm);
end


      function [y] = multiply_B(x, A,AT, scaling_factor,M,N)

        % Computes A(ATx). 
        % A is function handler to compute Ax
        % AT is a function handler ATx

        y1 = AT(x,scaling_factor,M/scaling_factor,N/scaling_factor);
        y = A(y1,scaling_factor,M,N);

        
      end


      function [x2] = conjgrad(A2, b, x2)
        % Implements the conjugate gradient method.
        % A2 is a function handler
        
        MAX_ITER = 1e6;
        TOL      = 10e-7;
        
        r = b - A2(x2, A, AT, scaling_factor, M, N);
        f = r;
        rsold = r'*r;
        
        for i = 1 : MAX_ITER
            
            Ap = A2(f, A, AT, scaling_factor, M, N);
            alpha = rsold/(f'*Ap);
            x2 = x2 + alpha*f;
            r = r - alpha*Ap;
            rsnew = r'*r;
            
            if sqrt(rsnew) < TOL
                break;
            end
         
            f = r + rsnew/rsold*f;
            rsold = rsnew;
        end

        if i == MAX_ITER
            fprintf('Conjugate gradient: Maximum number of iteration reached\n');
        end
        
    end
end









