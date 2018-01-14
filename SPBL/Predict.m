function testerr = Predict(Base_Para,W,TestObj)
% The output is [top1 err, top5 err].

mbase = length(Base_Para);
nSmptest = length(TestObj.gnd);

Htest = zeros(nSmptest,mbase);
% The i-th row of Htest is the scores of sample i by mbase base learners.
for j = 1:mbase
    switch Base_Para(j).type
        case 'tanh'
            Htest(:,j) = tanh(TestObj.fea*Base_Para(j).w + Base_Para(j).b);
        case 'logistic'
            Htest(:,j) = 1./(1+exp(-TestObj.fea*Base_Para(j).w-Base_Para(j).b));
        otherwise
            error('%d-th base learner type error in Predict: %s!',j,Base_Para(j).type);
    end
end
Score = Htest * W;
[~,predy] = max(Score,[],2);
testerr(1,1) = sum(predy~=TestObj.gnd)/nSmptest;% 0/0=NaN
[~,order] = sort(Score,2,'descend');
predy3 = order(:,1:3);%¸údÓÐ¹Ø
testerr(1,2) = 1 - sum(sum(predy3==repmat(TestObj.gnd,1,3),2))/nSmptest;




