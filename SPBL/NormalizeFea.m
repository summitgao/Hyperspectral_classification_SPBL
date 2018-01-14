function fea = NormalizeFea(fea,row)
% if row == 1, normalize each row of fea to have unit norm;
% if row == 0, normalize each column of fea to have unit norm;
%
%   version 3.0 --Jan/2012 
%   version 2.0 --Jan/2012 
%   version 1.0 --Oct/2003 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if ~exist('row','var')
    row = 1;
end

if row
    nSmp = size(fea,1);
    feaNorm = max(1e-14,full(sum(fea.^2,2)));
    fea = spdiags(feaNorm.^-.5,0,nSmp,nSmp)*fea;
else
    nSmp = size(fea,2);
    feaNorm = max(1e-14,full(sum(fea.^2,1))');
    fea = fea*spdiags(feaNorm.^-.5,0,nSmp,nSmp);
end
            
return;
