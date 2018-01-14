
clear all; close all; clc

load SeaIceDataset.mat;

addpath LBP_feat;
addpath L-BFGS-B-C;
addpath SPBL;

data0 = data./max(data(:)); % normalization
[m n d] = size(data0);

% band selection
X = reshape(data0, m*n, d);
Psi = PCA_Train(X',3); % 降维
X = X*Psi;
data0 = reshape(X, m, n, size(Psi,2));

% LBP feature extraction
fprintf(' ... ... LBP feature extraction begin ... ...\n');
%（半径，维度/采样点数）=（r，nr） 
r = 1;  nr = 8;
mapping = getmapping(nr,'u2'); 
% LBP块大小(2*W0+1 x 2*W0+1)
feat = LBP_feature_global(data0, r, nr, mapping, 10, gth);
fprintf(' !!!!!!! LBP feature extraction finished !!!!!!\n');


% save Labelled Data and the labels
no_class = max(gth(:));
data0 = reshape(data, m*n, d);

Data = []; Labels = [];
d = size(feat, 3);
Data_tmp = reshape(feat, m*n, d);
Data_tmp = [Data_tmp, data0];

for i = 1: no_class
    pos = find(gth==i);
    Data = [Data; Data_tmp(pos, :)];
    Labels = [Labels, length(pos)];
end
no_class = length(Labels);

% 这一行非常重要，指定了用于训练样本的数量
CTrain = [1000 1368 5269 1140];

DataTrn = [];
fprintf('   ... ... create Data used for training ... ...\n');
fprintf('   ... ... create Data used for testing  ... ...\n');
a = 0; 
for i = 1: no_class
    Data_tmp = Data((a+1):(Labels(i)+a), :);
    a = Labels(i) + a;
    rand('seed', 2);
    index_i = randperm(Labels(i));
    DataTrn = [DataTrn; Data_tmp(index_i(1:CTrain(i)), :)];
end


LabTrn = []; LabTst = [];
for i = 1: length(CTrain)
   LabTrn = [LabTrn; i * ones(CTrain(i),1)];
   LabTst = [LabTst; i * ones(Labels(i),1)];
end


clear Normalize Psi X a d;
clear i index_i m map mapping;
clear n no_class nr pos r z;
clear CTest CTrain;

% Set the training and validation sample proportion
train_ratio = 1.0;
vali_ratio = 0;
n_ratio = 0.0;
% maximum # of base learners:
maxBase = 180;

fea = DataTrn;
gnd = LabTrn;

fprintf('   ... ... Self-Paced Boost Training begin ... ...\n\n\n');
% run SPBL
Gen_Split;
SPBLtrain;
fprintf('\n\n\n   ... ... Self-Paced Boost Training finished !!! !!! \n\n\n');

fprintf('   ... ... Testing begin ... ...\n');

mbase = length(Base_Para);
nSmptest = length(LabTst);
for j = 1:mbase
    Htest(:,j) = 1./(1+exp(-Data*Base_Para(j).w-Base_Para(j).b));
end

Score = Htest * W;
[~,predy] = max(Score,[],2);
accu = 1-sum(predy~=LabTst)/nSmptest;

fprintf('   ... ... Testing accuracy is %f ... ...\n', accu);






















