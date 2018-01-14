function [u,dualgap,dualviolation,train_err] = DualTrans(H,W,traingnd,v,fval,niu,Wopts)

if nargin < 7 % 
    Wopts = struct('regutype','l1');
end

nSmp = size(H,1);
Score = H * W;
rou = bsxfun(@minus,Score((traingnd-1)*nSmp+(1:nSmp)'),Score);
% rou = bsxfun(@minus,sum(Score.*full(sparse(1:nSmp,traingnd,1)),2),Score);
[~,tr_predy] = max(Score,[],2);
train_err(1) = sum(tr_predy~=traingnd)/nSmp;
[~,tmporder] = sort(Score,2,'descend');
train_err(1,2) = 1 - sum(sum(tmporder(:,1:3)==repmat(traingnd,1,3),2))/nSmp;

% u = 1./(1+exp(rou));
u = bsxfun(@rdivide,v,1+exp(rou));
posu = u(u>0); posv = v(v>0 & v<1);
vu = bsxfun(@minus,v,u); pos_vu = vu(vu>0);
dualfval = size(W,2)*sum(posv.*log(posv))-sum(posu.*log(posu))-sum(pos_vu.*log(pos_vu));
dualgap =  1-dualfval/fval;

Q = H'*(full(sparse(1:nSmp,traingnd,sum(u,2)))-u);
switch Wopts.regutype
    case 'l1'
        dualviolation = max(max(Q))/niu - 1;
    case {'l12','l21'}
        Q(Q<0) = 0;
        dualviolation = max(sqrt(sum(Q.*Q,2)))/niu - 1;
end
