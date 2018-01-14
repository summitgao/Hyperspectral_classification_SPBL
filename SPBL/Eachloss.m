function loss = Eachloss(W,H,gnd)

nSmp = length(gnd);
Score = H * W;
rou = bsxfun(@minus,Score((gnd-1)*nSmp+(1:nSmp)'),Score);

loss = mean(log(1+exp(-rou)),2);

