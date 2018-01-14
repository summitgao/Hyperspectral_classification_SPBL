function [v, lambda] = SPLreweighting(loss,rankratio,SPLopts)

if ~isempty(find(loss<0,1)), error('SPLfcn error: loss must be all >= 0'); end
if nargin < 3,
    SPLopts = struct([]);
end
if ~isfield(SPLopts,'Type')
    SPLopts.Type = 'hard';
end

nSmp = length(loss);
[sortedloss,order] = sort(loss,'ascend');
% rank = zeros(nSmp,1); rank(order) = 1:nSmp;
rankthres = round(nSmp*rankratio);
lambda = (sortedloss(rankthres)+sortedloss(min(rankthres+1,nSmp)))/2;
%¹«Ê½£¨8£©
switch SPLopts.Type
    case 'hard'
        v = ones(length(loss),1);
        v(loss>=lambda) = 0;
    case 'linear'
        v = max(1-loss/lambda,0);
    case {'logarithmic', 'log'}
        if lambda >= 1,
            error('SPLfcn error: lambda must < 1 for logarithmic case!');
        end
        v = zeros(length(loss),1); v(loss==0) = 1;
        log_softidx = (loss<lambda & loss>0);
        v(log_softidx) = log(loss(log_softidx)+1-lambda)/log(1-lambda);
    
    % For mixture case, zeta is another needed SPL parameter known as 1/(k'-k)
    case {'mixture','mixture1','mix','mix1'}
        checkfield(SPLopts,'SPLopts','zeta',0);
        zeta = SPLopts.zeta * lambda;
        v = ones(length(loss),1); v(loss>=lambda) = 0;
        mix_softidx1 = (loss>zeta*lambda/(zeta+lambda) & loss<lambda);
        v(mix_softidx1) = zeta./loss(mix_softidx1) - zeta/lambda;
    case 'mixture2'
        checkfield(SPLopts,'SPLopts','zeta',0);
        zeta = SPLopts.zeta * lambda;
        lambda = sqrt(lambda);
        v = ones(length(loss),1); v(loss>=lambda^2) = 0;
        mix_softidx2 = (loss>(zeta*lambda/(zeta+lambda))^2 & loss<lambda^2);
        v(mix_softidx2) = zeta./sqrt(loss(mix_softidx2)) - zeta/lambda;
end
if isempty(find(v>0,1))
    error('SPLfcn error: all v == 0. lambda should be increased.');
end

end




