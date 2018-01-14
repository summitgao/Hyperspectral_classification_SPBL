function ini_idx = Inisel(gnd,ratio)

label = unique(gnd);
ini_idx = [];
for i = label'
    can_idx = find(gnd==i);
    ndoc = length(can_idx);
    ini_idx = [ini_idx;can_idx(sort(randperm(ndoc,round(ndoc*ratio)),'ascend'))];
end

