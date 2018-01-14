function [W,fval] = UpdateW(H,TrainObj,v,niu,Wopts)

bfgs_opts = struct('m',10, 'printEvery',Inf);
lb = zeros(size(H,2)*TrainObj.nClass,1); ub = Inf(size(lb));
[W,fval] = lbfgsb(@(W)Wobjfcn(H,W,TrainObj.gnd,v,niu,Wopts),lb,ub,bfgs_opts);
W = reshape(W,[],TrainObj.nClass); % W(W<1e-6) = 0;