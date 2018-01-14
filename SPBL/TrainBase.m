function [new_Base,newH] = TrainBase(trainfea,traingnd,u,options)

% Function TrainBase accepts the samples' weights 'u' and their labels
% 'traingnd', and generates the current weak learner, with a type specified in
% 'options'.
% trainfea - the N x D input matrix, where each row data(i,:) corresponds to a data sample

checkfield(options,'options','type');
checkfield(options,'options','regu_weight',0);
if ~isempty(find(u<0,1)), error('TrainBase error: u must all >= 0.'); end

addpath L-BFGS-B-C;

[nSmp,dFea] = size(trainfea);
[label,~,ic] = unique(traingnd);
nClass = length(label);
if norm(label-(1:nClass)') > 0
    labelidx = 1:nClass;
    traingnd = labelidx(ic);
    label = (1:nClass)';
end

augX = [trainfea,ones(nSmp,1)];

% bfgs_opts = struct('factr',1e5, 'pgtol',1e-7, 'm',10, 'printEvery',100);
bfgs_opts = struct('m',10, 'printEvery',Inf);
rw = options.regu_weight;
if nClass == 2 && size(u,2) == 1 % For the binary classification case
    y = ones(nSmp,1); y(traingnd==label(2)) = -1;
    lb = -Inf(dFea+1,1); ub = Inf(dFea+1,1);
    switch options.type
        case 'tanh'
            [OptAugw,Optfval,Info] =...
                lbfgsb(@(augw)tanhcost(augw,augX,u.*y,rw),lb,ub,bfgs_opts);
            newH = tanh(augX*OptAugw);
        case 'logistic'
            [OptAugw,Optfval,Info] =...
                lbfgsb(@(augw)logicost(augw,augX,u.*y,rw),lb,ub,bfgs_opts);
            newH = 1./(1+exp(-augX*OptAugw));
        otherwise
            error('Base learner type error in TrainBase: %s.',options.type);
    end
    Optfval = -Optfval;
    new_Base = struct('w',OptAugw(1:(end-1)),'b',OptAugw(end),...
        'fval',Optfval,'regu_weight',rw,'Info',Info,'type',options.type);
else % For the multiclass case
    if size(u,2) ~= nClass
        error('Dimension mismatch in TrainBase: size(u,2) ~= nlabel');
    end
    uy = full(sparse(1:nSmp,traingnd,sum(u,2))) - u;
    lb_en = -Inf((dFea+1)*nClass,1); ub_en = Inf(size(lb_en));
    switch options.type
        case 'tanh'
            [OptAugw,OptSumScore,~] = lbfgsb(@(augw)tanhcost(augw,augX,uy,rw),...
                lb_en,ub_en,bfgs_opts);
            OptSumScore = -OptSumScore;
            OptAugw = reshape(OptAugw,dFea+1,nClass);
            AllnewH = tanh(augX*OptAugw);
            
            OptScores = sum(AllnewH.*uy,1) - sum(OptAugw(1:end-1,:).^2,1)*rw/2;
            if abs(1-sum(OptScores)/OptSumScore) > 1e-5
                error('tanhcost objective error!');
            end
        case 'logistic'
            [OptAugw,OptSumScore,~] = lbfgsb(@(augw)logicost(augw,augX,uy,rw),...
                lb_en,ub_en,bfgs_opts);
            OptSumScore = -OptSumScore;
            OptAugw = reshape(OptAugw,dFea+1,nClass);
            AllnewH = 1./(1+exp(-augX*OptAugw));
            OptScores = sum(AllnewH.*uy,1) - sum(OptAugw(1:end-1,:).^2,1)*rw/2;
            if abs(1-sum(OptScores)/OptSumScore) > 1e-5
                error('logistic objective error!');
            end
        otherwise
            error('Base learner type error in TrainBase: %s.',options.type);
    end
    if isfield(options,'EachAug') && options.EachAug > 1
        [~,scoreorder] = sort(OptScores,'descend');
        for i = 1:options.EachAug
            idx = scoreorder(i);
            new_Base(i,1) = struct('w',OptAugw(1:(end-1),idx),'b',OptAugw(end,idx),...
                'maxScore',OptScores(idx),'regu_weight',rw,'maxClass',label(idx),...
                'type',options.type);
        end
        newH = AllnewH(:,scoreorder(1:options.EachAug));
    else
        [maxScore,maxidx] = max(OptScores);
        new_Base = struct('w',OptAugw(1:(end-1),maxidx),'b',OptAugw(end,maxidx),...
            'maxScore',maxScore,'regu_weight',rw,'maxClass',label(maxidx),...
            'type',options.type);
        newH = AllnewH(:,maxidx);
    end
    
end
end

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
end

function [cost,grad] =  tanhcost(augw,augX,uy,wei_regu)

augw = reshape(augw,[],size(uy,2));

cost_each = tanh(augX * augw);
cost = -sum(sum(cost_each.*uy)) + sum(sum(augw(1:end-1,:).^2))*wei_regu/2;

tanh_grad = 1-cost_each.*cost_each;
grad = -augX' * (tanh_grad.*uy) + [wei_regu*augw(1:end-1,:);zeros(1,size(uy,2))];
grad = reshape(grad,[],1);
end

