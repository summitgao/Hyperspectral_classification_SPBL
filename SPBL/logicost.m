function [cost,grad] =  logicost(augw,augX,uy,wei_regu)
% This function returns the logistic score and its grad w.r.t. the weights uy.
% augX: N x (D+1) matrix, where each row is a sample, augmented by 1 for each sample.
% augw: each column as [w_r;b_r], (D+1)xc matrix.
% uy = delta*sum(u,2)-u
% wei_regu: weight of l2-norm regularization of augw

% if nargin < 4
%     wei_regu = 1e-4;
% end

augw = reshape(augw,[],size(uy,2));

cost_each = 1./(1+exp(-augX*augw));
cost = -sum(sum(cost_each.*uy)) + sum(sum(augw(1:end-1,:).^2))*wei_regu/2;

logi_grad = cost_each.*(1-cost_each);
grad = -augX' * (logi_grad.*uy) + [wei_regu*augw(1:end-1,:);zeros(1,size(uy,2))];
grad = reshape(grad,[],1);