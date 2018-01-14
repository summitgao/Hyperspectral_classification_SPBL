

nAll = length(gnd);
label = unique(gnd);
nlabel = length(label);
ndoc = zeros(nlabel,1); docidx = cell(nlabel,1);
for i = 1:nlabel
    docidx{i} = find(gnd == label(i));
    ndoc(i) = length(docidx{i});
end
ntrain = round(ndoc * train_ratio);
nvali = round(ndoc * vali_ratio);
ntest = ndoc - ntrain - nvali;
trainidx = []; valiidx = []; testidx = [];
for i = 1:nlabel
    randidx = randperm(ndoc(i))';
    trainidx = [trainidx; sort(docidx{i}(randidx(1:ntrain(i))),'ascend')];
    valiidx = [valiidx; sort(docidx{i}(randidx(ntrain(i)+(1:nvali(i)))),'ascend')];
    testidx = [testidx; sort(docidx{i}(randidx(ntrain(i)+nvali(i)+1:ndoc(i))),'ascend');];
end
if length(unique([trainidx;valiidx;testidx])) ~= nAll
    error('Train and test indices error!');
end
save(['Data/Split',num2str(train_ratio*100),'.mat'],...
    'trainidx','valiidx','testidx','train_ratio','vali_ratio','ndoc','docidx');
