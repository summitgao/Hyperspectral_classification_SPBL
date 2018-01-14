clear;
%% Set the label noise ratio
n_ratio = 0.15;

%% Randomly select the noisily labeled samples, evenly in each class
load('Data/Split50.mat','trainidx','train_ratio');
load('Data/fea.mat', 'gnd');

nSmp = length(trainidx);
traingnd = gnd(trainidx);
label = unique(traingnd); nlabel = length(label);
ntrain = zeros(nlabel,1);
for i=1:nlabel, ntrain(i) = sum(traingnd==label(i)); end
n_traingnd = traingnd;
noiseidx = [];
startidx = 0;
for i = 1:nlabel
    randidx = randperm(ntrain(i),round(ntrain(i)*n_ratio));
    randidx = sort(randidx,'ascend')+startidx;
    noiseidx = [noiseidx;randidx'];
    for j = randidx
        tmp = label([1:i-1,i+1:end]);
        n_traingnd(j) = tmp(randperm(nlabel-1,1));
    end
    startidx = startidx + ntrain(i);
end
if norm(find(n_traingnd~=traingnd)-noiseidx)~=0, error('noiseidx error!'); end
fprintf(1,'noise ratio = %.4f\n',length(noiseidx)/nSmp);

save(['Data/n',num2str(n_ratio*100),'_traingnd.mat'],...
    'n_traingnd','trainidx','train_ratio','n_ratio','noiseidx');
