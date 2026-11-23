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

rng(1)
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

m_tr = mean(tr_data);
std_tr = std(tr_data);
% 데이터 정규화
ntr_data = [];
for kk=1:1:4
    ntr_data(:,kk) = (tr_data(:,kk) - m_tr)/std_tr;
end
% 4개 특징을 이용한 모델링
Mdl = TreeBagger(50, ntr_data, tr_label, Method="classification");

[est_label_f4] = predict(Mdl, ntr_data);
est_label_f4 = cell2mat(est_label_f4);
est_label_f4 = str2num(est_label_f4);

figure;
confusionchart(tr_label, est_label_f4);

figure;
subplot(211); bar(est_label_f4); axis tight;
subplot(212); bar(tr_label); axis tight;


% PCA
[coeff,tr_score,latent,tsquared,explained,mu] = pca(ntr_data);

idx1 = find(tr_label == 1);
idx2 = find(tr_label == 2);
idx3 = find(tr_label == 3);

figure;
subplot(221); 
plot(ntr_data(idx1, 1), ntr_data(idx1, 2), 'ro'); hold on; 
plot(ntr_data(idx2, 1), ntr_data(idx2, 2), 'go'); hold on; 
plot(ntr_data(idx3, 1), ntr_data(idx3, 2), 'bo'); hold off; 
title('특징1 vs. 특징2');
subplot(222); 
plot(ntr_data(idx1, 3), ntr_data(idx1, 4), 'ro'); hold on; 
plot(ntr_data(idx2, 3), ntr_data(idx2, 4), 'go'); hold on; 
plot(ntr_data(idx3, 3), ntr_data(idx3, 4), 'bo'); hold off; 
title('특징3 vs. 특징4');
subplot(223); 
plot(tr_score(idx1, 1), tr_score(idx1, 2), 'ro'); hold on; 
plot(tr_score(idx2, 1), tr_score(idx2, 2), 'go'); hold on; 
plot(tr_score(idx3, 1), tr_score(idx3, 2), 'bo'); hold off; 
title('pc1 vs. pc2');
subplot(224); 
plot(tr_score(idx1, 3), tr_score(idx1, 4), 'ro'); hold on; 
plot(tr_score(idx2, 3), tr_score(idx2, 4), 'go'); hold on; 
plot(tr_score(idx3, 3), tr_score(idx3, 4), 'bo'); hold off; 
title('pc3 vs. pc4');

ts_data = meas(ts_id,:);
nts_data = [];
for kk=1:1:4
    nts_data(:,kk) = (ts_data(:,kk) - m_tr)/std_tr;
end
ts_scores = nts_data*coeff;

ts_label = spcs2num(ts_id,:);


% random forest 생성. 예. 나무 50개
Mdl = TreeBagger(50, tr_score(:,1:2), tr_label, Method="classification");

view(Mdl.Trees{1},Mode="graph");

[est_label] = predict(Mdl, ts_scores(:,1:2));
est_label = cell2mat(est_label);
est_label = str2num(est_label);

figure;
confusionchart(ts_label, est_label);

figure;
subplot(211); bar(est_label); axis tight;
subplot(212); bar(ts_label); axis tight;



