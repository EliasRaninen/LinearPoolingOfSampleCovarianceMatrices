function [Sigmas, A, params, QPflag] = linpool(DataCell,method,identity,onlyfirst,aI_LB)
% LINPOOL computes regularized sample covariance matrix estimates by pooling
% the class SCMs (and possibly the identity matrix) via a non-negative
% linear combination as explained in E. Raninen, D. E. Tyler, and E. Ollila,
% "Linear pooling of sample covariance matrices", arXiv preprint,
% arXiv:2008.05854, 2021.
%
% Each class covariance matrix estimate is:
%                K
%  Sigmas{k} =  SUM A(j,k)*SCM{j} + A(K+1,k)*eye(p),
%               j=1
%
%  where A(K+1,k) >= aI_LB  (default value aI_LB = 1e-8)
%
% IMPLEMENTATION:
%   The function first computes the unconstrained solution for the optimal
%   coefficients by using Matlab's '\' operator (mldivide.m, backslash, left
%   matrix divide). If the optimal coefficients have negative values, the
%   function imposes the nonnegativity constraint and uses Matlab's QP
%   solver quadprog.m.
%
% USAGE:
%
% [Sigmas, A, params, QPflag] = linpool(DataCell,method,identity,onlyfirst,aI_LB)
%
% [Sigmas, A, params, QPflag] = linpool(DataCell)
% [Sigmas, A, params, QPflag] = linpool(DataCell,'convex')
% [Sigmas, A, params, QPflag] = linpool(DataCell,[],false)
% [Sigmas, A, params, QPflag] = linpool(DataCell,[],true,true)
%
% INPUT:
%       DataCell  - a Kx1 cell array of data. Each cell is a (n_k x p)
%                   matrix. The data can be real or complex-valued.
%       method    - 'linear' or 'convex'. Default is 'linear'. Choose
%                   whether to compute a nonnegative linear or a convex
%                   combination of the SCMs.
%       identity  - true or false (logical). Default is true. When true,
%                   shrinkage towards the identity matrix is included.
%       onlyfirst - true or false (logical). Default is false. When true,
%                   the estimator is computed only for the first class
%                   (k=1).
%       aI_LB     - minimum value for the coefficient corresponding to the
%                   identity matrix, i.e., A(K+1,k) >= aI_LB(k). By default
%                   aI_LB = 1e-8. In low sample size cases, when A(K+1,k)
%                   is close to zero there may be a possibility that the
%                   estimate is not well-conditioned despite having a low
%                   MSE. This can result in high error when inverting the
%                   estimate. In these cases it may be useful to increase
%                   the lower bound.
%                    
%
% OUTPUT:
%       Sigmas    - a Kx1 cell array of computed covariance matrix
%                   estimates.
%       A         - a (K+1)xK matrix of coefficients used in the linear
%                   combination.
%       SCM       - a Kx1 cell array of computed SCMs.
%       params    - a stuct of computed statistical parameters.
%       QPflag    - is true (logical) if QP solver was used for at least
%                   one class.

% Here,
%  K   : the total number of classes,
%  n_k : the number of samples,
%  p   : the dimension of the samples.
%
%
% UPDATE HISTORY:
%   Aug 13, 2020:
%       - initial version.
%   Oct  2, 2020 and Nov  3, 2020: 
%       - cleaned spatialmedian.m (removed assert functions).
%   Dec 19, 2020:
%       - updated to support complex-valued data. Also moved all functions
%         to the same file linpool.m.
%   Aug 20, 2021:
%       - added possibility to determine a lower-bound (aI_LB) for the
%         shrinkage intensity of the identity matrix. In certain cases (not
%         always), increasing the lower-bound can reduce the error, when
%         inverting the matrix.
%       - fixed a bug related to ensuring that the theoretical lower-bound
%         of the kurtosis is respected. In the updated code, if the
%         estimated kurtosis is equal or less than the theoretical
%         lower-bound, it is set to 0.99 * theoretical lower bound.
%       - added update history.
%
% By E. Raninen and E. Ollila (2021)

verbose = false;
K = numel(DataCell); % number of classes
p = size(DataCell{1},2); % dimension
QPflag = false;

% Include identity shrinkage by default
if nargin < 3 || isempty(identity)
    identity = true;
end

% Specify for which classes the estimator is computed
if nargin < 4 || isempty(onlyfirst)
    onlyfirst = false;
end
if onlyfirst
    G = 1;
else
    G = 1:K;
end

if ~exist('method','var') || isempty(method)
    method = 'linear';
end
if strcmp(method,'convex')
    linear = false;
else
    linear = true;
end


opts = optimoptions('quadprog','Display','Off');

% Compute the parameter matrices C and D
[C,D,SCM,params] = get_C_and_D(DataCell);
params.C = C;
params.SCM = SCM;

if identity % if shrinkage towards the identity is included
    if nargin < 5 || isempty(aI_LB)
        aI_LB = 1e-8*ones(1,K);
    elseif length(aI_LB) == 1
        aI_LB = aI_LB(1)*ones(1,K);
    elseif numel(aI_LB) == K
        aI_LB = aI_LB(:).';
    else
        error('linpool.m: value of aI_LB is not valid.');
    end
    C = [C params.eta; params.eta.' 1];
    D = [D zeros(K,1); zeros(1,K) 0];
end

% Compute the weight matrix A
if linear % non-negative linear combination
    A = (C + D)\C(:,G);
    A(abs(A(:))<eps) = 0;

    % use QP solver quadprog.m if there are negative coefficients or if the
    % coefficient for the identity is not positive
    neg_ind = any(A<0); % negative coefficients
    if identity
        force_identity_shrinkage_ind = (A(K+1,G) < aI_LB(G));
    else
        force_identity_shrinkage_ind = false(1,numel(G));
    end
    
    QP_ind = or(neg_ind,force_identity_shrinkage_ind);
    lb = zeros(size(C,1),1); % lower bound of coefficients
    if any(QP_ind)
        QPflag = true;        
        for k = G(QP_ind)
            % solve coefficients of linear combination
            if verbose
                fprintf('Running the QP solver with positivity constraint of class %d.\n',G(k));
            end
            if identity
                lb(end) = aI_LB(k);
            end
            % QP with positivity constraint
            A(:,k) = quadprog(D + C,-C(:,k),[],[],[],[],lb,[],[],opts);
        end
    end

else % convex combination
    QPflag = true;
    % imposing the constaint that weights are positive and sum to 1 for each class
    A = nan(size(C(:,G)));
    
    lb = zeros(size(C,1),1); % lower bound of coefficients

    for k = G
        if identity
            lb(end) = aI_LB(k);
        end
        A(:,k) = quadprog(D + C,-C(:,k),[],[],ones(1,size(C,1)),1,lb,[],[],opts);
    end
end

% Compute covariance matrix estimates based on A
Sigmas = cell(numel(G),1);
for k = G
    if identity  
        Sigmas{k} = A(K+1,k)*eye(p);
    else
        Sigmas{k} = zeros(p);
    end
    for j=1:K
        Sigmas{k} = Sigmas{k} + A(j,k)*SCM{j};
    end
end

end


%% Auxiliary functions

function [C,D,SCM,params] = get_C_and_D(DataCell)
% GET_C_AND_D estimates of the parameter matrices C and D from the data.
% Auxiliary function that is needed by LINPOOL
%
% INPUT:
%    DataCell  - a Kx1 cell array of data. Each cell is a (n_k x p) matrix.
%
% OUTPUT:
%    C      - matrix of size K x K of estimates of tr(Sigma_i Sigma_j)/p
%    D      - KxK diagonal matrix whose diagonal elements are estimates of
%             MSE(S_i), i = 1,...,K.
%    SCM    - a Kx1 cell array of sample covariance matrices.
%    params - struct of estimates of parameters
%
% By E. Raninen and E. Ollila (2020)

assert(size(DataCell,2)==1);

% number of classes
K = numel(DataCell);
SCM   = cell(K,1); % sample covariance matrices (SCMs)
SSCM  = cell(K,1); % spatial SCMs
trSk  = nan(K,1);  % scale estimates tr(S_k)/p
gam   = nan(K,1);  % estimates of sphericity
kappa = nan(K,1);  % estimates of elliptical kurtosis
p     = size(DataCell{1},2);
n     = nan(K,1);

for k=1:K
    Xk = DataCell{k};
    n(k) = size(Xk,1);
    SCM{k} = conj(cov(Xk));
    kappa(k) = estimate_kurt(Xk);
    trSk(k) = real(trace(SCM{k}));
end
eta = trSk/p;

% compute sphericity estimate
for k=1:K
    [SSCM{k},~,d] = SpatialSCM(DataCell{k});
    gam(k) = estimate_sphericity(SSCM{k},d);
end

% Estimate of tr(Sigma_k^2)
tr_Ck2 = p*eta.^2.*gam;

% Estimate of tr(Sigma_k)^2
trCk_2 = trSk.^2;

% Compute D
if isreal(DataCell{1}) % for real-valued data
    tau1 = 1./(n-1)+kappa./n;
    tau2 = kappa./n;
    MSE_Sk = tau1.*trCk_2 + (tau1+tau2).*tr_Ck2;
    D = diag(MSE_Sk/p);
else % for complex-valued data
    tau1 = 1./(n-1)+kappa./n;
    tau2 = kappa./n;
    MSE_Sk = tau1.*trCk_2 + tau2.*tr_Ck2;
    D = diag(MSE_Sk/p);
end
    
% Compute inner product matrix  C = [tr(Sigma_i x Sigma_j)]
C = zeros(K);
C(1:(K+1):K^2) = tr_Ck2;
% Compute off-diagonals based on spatial sign covariance matrix
trCitrCj = (trSk*trSk');
for k=1:(K-1)
   for j=(k+1):K
      C(k,j)  = trace(SSCM{k}*SSCM{j})*trCitrCj(k,j);
      C(j,k)  = C(k,j);
   end
end
C = real(C);
C = C/p;

% Save compute values
params.SSCM = SSCM; % spatial sign covariance matrix
params.eta = eta;
params.kappa = kappa;
params.gam = gam;
params.diagD = diag(D);
params.trCk_2 = trCk_2;
params.tr_Ck2 = tr_Ck2;
end

%%
function [gammahat,gammahat0] = estimate_sphericity(SSCM,d)
% Computes the estimate for the sphericity for a given SSCM
%
% INPUTS:
%
%   SSCM             spatial sign covariance matrix of size n x p
%                    (rows are observations), can be complex or real-valued.
%   d                Euclidean lengths of (centered) observations.
%
% Optional inputs:
%
%   is_centered     (logical) is the X already centered. Default=false.
%   muhat           p x 1 vector (e.g., spatial median of the data)
%
% OUTPUTS:
%
%   gammahat        Estimator of sphericity. The sphericity estimator uses
%                   a correction factor developed in C. Zou et. al.
%                   "Multivariate sign-based high-dimensional tests for
%                   sphericity,” Biometrika, vol. 101, no. 1, pp. 229–236,
%                   2014, that improves gammahat0 estimator when p/n is
%                   large.
%
%   gammahat0       This is the estimator of sphericity without the bias
%                   correction
%
% By E. Raninen and E. Ollila (2020)

p = size(SSCM,1);
assert(size(SSCM,1)==size(SSCM,2));
n = numel(d);

m3 = mean(d.^(-3));
m2 = mean(d.^(-2));
m1 = mean(1./d);
ratio = m2/(m1^2);
ratio3 = m3/(m1^3);
delta = (1/n^2)*(2 - 2*ratio + ratio^2) + ...
    (1/n^3)*(8*ratio - 6*ratio^2 + 2*ratio*ratio3 - 2*ratio3);

gammahat0 = (p*n/(n-1))*(trace(SSCM^2) - 1/n);
gammahat0 = real(gammahat0);
gammahat  = gammahat0 - p*delta;

% NOTE:\gamma in [1, p];
gammahat0 = min(p,max(1,gammahat0));
gammahat  = min(p,max(1,gammahat));
end

%%
function [kappahat, xbar] = estimate_kurt(X,xbar,is_centered,print_info)
% ESTIMATE_KURT computes the estimate of the elliptical kurtosis parameter
% of a p-dimensional distribution given the data set X.
%
% [kappahat, xbar] = estimate_kurt(X,...)
%
% INPUTS:
%   X               data matrix of size n x p (rows are observations). Can
%                   be real or complex-valued. 
% OPTIONAL INPUTS:
%   xbar            sample mean vector of the data X.
%   is_centered     (logical) is the X already centered. Default=false.
%   print_info      (logical) verbose flag. Default=false.
%
% Modified from ellkurt.m of the toolbox:
% RegularizedSCM available from http://users.spa.aalto.fi/esollila/regscm/
%
% By E. Ollila and E. Raninen (2020)

[n,p] = size(X);

if isreal(X) 
    ka_lb = -2/(p+2); % theoretical lower bound for the kurtosis parameter
else 
    ka_lb = -1/(p+1); % theoretical lower bound for kurtosis parameter
end

if nargin < 4 || isempty(print_info)
    print_info = false;
end

if nargin < 3 || isempty(is_centered)
    is_centered = false;
end

if nargin < 2 || isempty(xbar)
    xbar = mean(X);
end

if ~is_centered
    if print_info, fprintf('estimate_kurt: centering the data...'); end
    X = X - repmat(xbar,n,1);
end
vari =  mean(abs(X).^2);

indx = (vari==0);
if any(indx)
    if print_info
        fprintf('estimate_kurt: found a variable with a zero sample variance\n');
        fprintf('         ...ignoring the variable in the calculation\n');
    end
end

if isreal(X) 
    kurt1n = (n-1)/((n-2)*(n-3));
    g2 = mean(X(:,~indx).^4)./(vari(~indx).^2)-3;
    G2 = kurt1n*((n+1)*g2 + 6);
    kurtest = mean(G2);
    kappahat = (1/3)*kurtest;
else
    g2 = mean(abs(X(:,~indx)).^4)./(vari(~indx).^2)-2;
    kurtest = mean(g2);
    kappahat = (1/2)*kurtest;
end

if kappahat > 1e6
    error('estimate_kurt: something is wrong, too large value for kurtosis\n');
end

% kappahat has to be strictly larger than ka_lb
if kappahat <= ka_lb
      kappahat = 0.99*ka_lb;
end
end

%%
function [Csgn,muhat,d] = SpatialSCM(X,is_centered,muhat)
% SPATIALSCM computes the spatial sign covariance matrix.
%
% INPUTS:
%
%   X               data matrix of size n x p (rows are observations)
%                   can be complex-valued or real-valued data.
% OPTIONAL INPUTS:
%
%   is_centered     (logical) is the X already centered. Default=false.
%   muhat           1 x p vector (e.g., spatial median of the data) to be
%                   used for centering the data.
%
% OUTPUTS:
%
%   Csgn            Spatial sign covariance matrix (p x p matrix)
%   muhat           Spatial median (1 x p vector) or the given muhat vector
%   d               Euclidean lengths of (centered) observations.
%
% By E. Ollila and E. Raninen (2020)

print_info = false;
p = size(X,2);

if nargin < 2 || isempty(is_centered)
    is_centered = false;
end

if ~is_centered
    if print_info
        fprintf('centering the data');
    end
    if nargin < 3
        muhat = spatmed(X);
    else
        assert(isequal(size(muhat),[1 p]));
    end
    X = bsxfun(@minus,X,muhat);
else
  if print_info
      fprintf('Not centering the data');
  end
end

d = sqrt(sum(X.*conj(X),2));
X = X(d~=0,:); % eliminate observations that have zero length
n = size(X,1);
X = bsxfun(@rdivide, X,d(d~=0));
Csgn = X'*X/n; % Sign covariance matrix
d(d<1.0e-10)= 1.0e-10;
end

%%
function  smed = spatmed(X,print_info)
% SPATMED computes the spatial median of the data set X.
%
% INPUTS:
%   X               data matrix of size n x p (rows are observations)
%                   Can be complex- or real-valued. 
% OPTIONAL INPUTS:
%   print_info      (logical) verbose flag. Default=false
%
% modified from toolbox:
% RegularizedSCM available from http://users.spa.aalto.fi/esollila/regscm/
%
% By E. Ollila and E. Raninen (2020)

if nargin ==1
    print_info = false;
end

if ~islogical(print_info)
    error('Input ''print_info'' needs to be logical');
end

len = sum(X.*conj(X),2); 
X = X(len~=0,:);
n = size(X,1);

if isreal(X)
    smed0 = median(X);
else
    smed0 = mean(X);
end
norm0 = norm(smed0);

iterMAX = 500;
EPS = 1.0e-4;
%TOL = 1.0e-6;
TOL = 1.0e-10;

for iter = 1:iterMAX

   Xc = bsxfun(@minus,X,smed0);
   len = sqrt(sum(Xc.*conj(Xc),2)); 
   len(len<EPS)= EPS;
   Xpsi = bsxfun(@rdivide, Xc, len);
   update = sum(Xpsi)/sum(1./len);
   smed = smed0 + update;

   dis = norm(update)/norm0;
   %fprintf('At iter = %3d, dis=%.6f\n',iter,dis);

   if (dis<=TOL)
       break;
   end
   smed0 = smed;
   norm0 = norm(smed);

end

if print_info
   fprintf('spatmed::convergence at iter = %3d, dis=%.10f\n',iter,dis);
end
end
