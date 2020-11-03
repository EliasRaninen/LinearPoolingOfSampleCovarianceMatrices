function  smed = spatialmedian(X,n,p)
%SPATIALMEDIAN(X,n,p) Computes the spatial median of the samples. 
% Inputs:
%           X - n times p matrix of samples,
%           n - number of samples,
%           p - dimension of samples.
% 
% By E. Ollila. (Edited by E. Raninen)

iterMAX = 2000;
EPS = 1.0e-4;
TOL = 1.0e-14;


assert(all(size(X) == [n p]))

len = sum(X.^2,2); 
X = X(len~=0,:);

smed0 = median(X);
norm0 = norm(smed0);

for iter = 1:iterMAX 

   Xc = bsxfun(@minus,X,smed0);
   len = sqrt(sum(Xc.^2,2));
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

% check zero gradient equation
%meangrad = mean(abs(sum(Xpsi)));

if iter == iterMAX
    fprintf('Spatialmedian.m : slow convergence, iterMAX reached. dis: %e, TOL: %e\n',dis,TOL)
end

