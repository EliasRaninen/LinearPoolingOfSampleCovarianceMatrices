function params = estimate_parameters(dataFromClasses)
%%ESTIMATE_PARAMETERS estimates parameters and computes statistics from the
%given data.
% input:
%  dataFromClasses - a (number of classes x 1) cell of 
%                    (samples x dimension) data matrices.
% output:
%  params - struct of estimates of parameters, where
%
%  * Sk, Si, or Sj denotes the SCM of class k, i or j,
%  * Ck, Ci, or Cj denotes the estimated true covariance matrix of class k, i, or j,
%  * S (e.g., in EtrSkS) denotes the pooled SCM:
%       (1/total number of samples) * sum_i (number of samples in class i) x Si
%  * tr_Ck2 means trace(Ck^2)
%  * trCk_2 means trace(Ck)^2
%
% By E. Raninen (2020)

assert(size(dataFromClasses,2)==1);

%% estimate statistical parameters

% number of classes
K = size(dataFromClasses,1);

SCM             = cell(K,1);
SSCM            = cell(K,1);
eta             = nan(K,1);
gam             = nan(K,1);
kappa           = nan(K,1);
n               = nan(K,1);

for k=1:K
    % data matrix of class k
    Xk = dataFromClasses{k};
    assert(isreal(Xk))
    
    % number of samples from classes
    [n(k), p] = size(Xk);
    
    % sample covariance matrices (normalization by (n(k)-1))
    SCM{k} = cov(Xk);

    % spatial sign covariance matrix
    SSCM{k} = compute_SSCM(Xk,n(k),p);
  
    % estimate scale
    eta(k) = trace(SCM{k})/p;
    
    % estimate sphericity
    gam(k) = min(p,max(1,(n(k)/(n(k)-1))*(p*trace(SSCM{k}^2) - p/n(k))));
    
    % estimate elliptical kurtosis
    kappa(k) = mean(max(kurtosis(Xk)/3-1, -2/(p+2)));
end

% proportion of samples from each class
PI = n./sum(n);

% pooled sample covariance matrix
S  = zeros(p);
for k=1:K
    S = S + PI(k)*SCM{k};
end

% tr(C_k^2)
tr_Ck2 = p*eta.^2.*gam;
% tr(C_k)^2
trCk_2 = eta.^2*p^2;

% tau 1 and tau 2
tau1 = 1./(n-1)+kappa./n;
tau2 = kappa./n;

% E[tr(Sk^2)]
Etr_Sk2 = tau1.*trCk_2 + (1+tau1+tau2).*tr_Ck2;
% E[tr(S_k)^2]
EtrSk_2 = 2*tau1.*tr_Ck2 + (1+tau2).*trCk_2;
% tr(var(vec(Sk)))
MSE_Sk = tau1.*(trCk_2 + tr_Ck2) + tau2.*tr_Ck2;

% compute inner products
trCiCj = zeros(K);
trCitrCj = zeros(K);
for k=1:K
    trCiCj(k,k) = tr_Ck2(k);
    trCitrCj(k,k) = trCk_2(k);
    jstart = k+1;
    if jstart <= K
        for j=jstart:K
            trCitrCj(k,j) = eta(k)*eta(j)*p^2;
            trCitrCj(j,k) = trCitrCj(k,j);
            
            trCiCj(k,j)  = trace(SSCM{k}*SSCM{j})*trCitrCj(k,j);
            trCiCj(j,k)  = trCiCj(k,j);
        end
    end
end
EtrSiSj = trCiCj;
EtrSiSj(logical(eye(K))) = Etr_Sk2;
EtrSitrSj = trCitrCj;
EtrSitrSj(logical(eye(K))) = EtrSk_2;

%% Save compute values
params.p = p;
params.K = K;
params.n = n;
params.PI = PI;
params.SCM = SCM;       % sample covariance matrix
params.SSCM = SSCM;     % spatial sign covariance matrix
params.S = S;           % pooled SCM
params.eta = eta;
params.kappa = kappa;
params.gam = gam;

PIiPIj = PI(:)*PI(:).';
PImat  = repmat(PI(:).',[K 1]);

params.MSE_Sk = MSE_Sk; % this is needed in the diagonal of matrix D of the linear pooling paper
params.trCiCj = trCiCj; % this is needed in matrix C of the linear pooling paper

params.trCitrCj = trCitrCj;
params.EtrSiSj = EtrSiSj;
params.EtrSitrSj = EtrSitrSj;
params.Etr_S2 = sum(sum((PIiPIj.*EtrSiSj)));
params.EtrS_2 = sum(sum((PIiPIj.*EtrSitrSj)));
params.EtrSkS = sum(PImat.*EtrSiSj,2);
params.EtrCkS = sum(PImat.*trCiCj,2);
params.EtrSktrS = sum(PImat.*EtrSitrSj,2);
params.EtrCktrS = sum(PImat.*trCitrCj,2);

