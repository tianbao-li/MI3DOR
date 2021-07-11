function [feature_fusion, label_fusion] = my_pooling(feature, label, view_num, flag)
%n * d -> n/view_num * d
%flag: 0 max 1 mean
[n, d] = size(feature);
feature_fusion = zeros(n/view_num, d);
label_fusion = zeros(1, n/view_num);
for i = 1:n/view_num
    tmp = feature((i-1)*view_num+1:i*view_num, 1:d);
    if flag
        feature_fusion(i, 1:d) = mean(tmp, 1);
    else
        feature_fusion(i, 1:d) = max(tmp, [], 1);
    end
    label_fusion(1,i) = label(1, (i-1)*view_num+1);
end
