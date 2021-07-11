% Retrieval
% load the features and labels of source and target domains
load('./Results_features_labels.mat')

source_label = source_label + 1;
target_label = target_label + 1;

% view pooling (if necessary)
% [target_feature, target_label] = my_pooling(target_feature, target_label, 12, 0);

target_feature_norm = normalize(target_feature);
source_feature_norm = normalize(source_feature);

% compute the simialrity matrix
final_adj=pdist2(source_feature_norm, target_feature_norm);

% compute the retrieval criteria
[pingce,pr_cure]=cross_performance(final_adj', target_label, source_label);

save('Results_pingce+prcure.mat','pingce','pr_cure');
disp(pingce);