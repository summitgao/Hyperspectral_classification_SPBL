function [loss,grad] = Wobjfcn(H,W,traingnd,v,niu,Wopts)

if nargin < 6
    Wopts = struct('regutype','l1');
end

[nSmp,mbase] = size(H);
W = reshape(W,mbase,[]);

Score = H * W;
rou = bsxfun(@minus,Score((traingnd-1)*nSmp+(1:nSmp)'),Score);
% rou = bsxfun(@minus,sum(Score.*Delta,2),Score);

%% Compute losses: allloss = log(1+exp(-rou));
all_loss = zeros(size(rou));
idx1 = find(rou<-10);
all_loss(idx1) = -rou(idx1) + log(exp(rou(idx1))+1);
idx2 = find(rou>=-10);
all_loss(idx2) = log(1+exp(-rou(idx2)));

loss = sum(sum(bsxfun(@times,all_loss,v)));

%% Compute Gradient
l_rou = -1./(1+exp(rou));
grad = bsxfun(@times,H',v')*( -l_rou + full(sparse(1:nSmp,traingnd,sum(l_rou,2))) );
switch Wopts.regutype
    case 'l1'
        loss = loss + niu * sum(sum(W));
        grad = grad + niu;
    case {'l12','l21'}
        normrow = sqrt(sum(W.*W,2));
        loss = loss + niu * sum(normrow);
        posidx = find(normrow>0);
        aux = W;
        aux(posidx,:) = bsxfun(@rdivide,W(posidx,:),normrow(posidx,:));
        grad = grad + niu * aux;
    otherwise
        error('Wopt.regutype not supported!');
end

grad = reshape(grad,[],1);
