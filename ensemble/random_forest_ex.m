clear; close all; clc;

load fisheriris

spcs2num = [];
for k=1:1:length(species)
    if strcmp(species(k), 'setosa') == 1
        spcs2num(k,1) = 1;
    elseif strcmp(species(k), 'versicolor') == 1
        spcs2num(k,1) = 2;
    elseif strcmp(species(k), 'virginica') == 1
        spcs2num(k,1) = 3;
    end
end

rng(3)
% training set, test set 나누기
% 각 그룹의 20%를 추출
[tr_idx1, ~, ts_idx1] = dividerand(1:1:50, 0.8, 0, 0.2);
[tr_idx2, ~, ts_idx2] = dividerand(51:1:100, 0.8, 0, 0.2);
[tr_idx3, ~, ts_idx3] = dividerand(101:1:150, 0.8, 0, 0.2);

ts_id = [];
ts_id = [ts_idx1 ts_idx2 ts_idx3];

tr_id = [];
tr_id = [tr_idx1 tr_idx2 tr_idx3];

tr_data = meas(tr_id,:);
tr_label = spcs2num(tr_id,:);

ts_data = meas(ts_id,:);
ts_label = spcs2num(ts_id,:);


% random forest 생성. 예. 나무 50개
Mdl = TreeBagger(50, tr_data, tr_label, Method="classification");

view(Mdl.Trees{1},Mode="graph");

[est_label] = predict(Mdl, ts_data);
est_label = cell2mat(est_label);
est_label = str2num(est_label);

figure;
confusionchart(ts_label, est_label);

figure;
subplot(211); bar(est_label); axis tight;
subplot(212); bar(ts_label); axis tight;


%% decision tree
mdl_dt = fitctree(tr_data, tr_label);
view(mdl_dt,'Mode','graph');

[dt_est] = predict(mdl_dt, ts_data);

figure;
subplot(211); bar(dt_est); axis tight;
subplot(212); bar(ts_label); axis tight;

figure;
confusionchart(ts_label, dt_est);
