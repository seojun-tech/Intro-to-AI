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
[tr_idx1, ~, ts_idx1] = dividerand(1:1:50, 0.85, 0, 0.15);
[tr_idx2, ~, ts_idx2] = dividerand(51:1:100, 0.85, 0, 0.15);
[tr_idx3, ~, ts_idx3] = dividerand(101:1:150, 0.85, 0, 0.15);

tr_id = [];
tr_id = [tr_idx1 tr_idx2 tr_idx3];
tr_label = spcs2num(tr_id,:);
tr_data = meas(tr_id,:);
% 학습 데이터의 평균과 표준편차 가지고 있기
tr_mean = mean(tr_data);
tr_std = std(tr_data);

% 학습 데이터 정규화
notr_data = [];
for kk=1:1:4
    notr_data(:,kk) = (tr_data(:,kk) - tr_mean(1, kk))/tr_std(1, kk);
end

ts_id = [];
ts_id = [ts_idx1 ts_idx2 ts_idx3];
ts_label = spcs2num(ts_id,:);
ts_data = meas(ts_id,:);

% 평가 데이터 정규화
nots_data = [];
for kk=1:1:4
    nots_data(:,kk) = (ts_data(:,kk) - tr_mean(1, kk))/tr_std(1, kk);
end


%%
% 4개 특징을 이용한 모델링: random forest
Mdl = TreeBagger(50, notr_data, tr_label, Method="classification");

[est_label_f4] = predict(Mdl, nots_data);
est_label_f4 = cell2mat(est_label_f4);
est_label_f4 = str2num(est_label_f4);

figure;
confusionchart(ts_label, est_label_f4);

figure;
subplot(211); bar(est_label_f4); axis tight;
subplot(212); bar(ts_label); axis tight;

%%
% PCA
[coeff,tr_score,latent,tsquared,explained,mu] = pca(notr_data);

idx1 = find(tr_label == 1);
idx2 = find(tr_label == 2);
idx3 = find(tr_label == 3);

figure;
subplot(221); 
plot(notr_data(idx1, 1), notr_data(idx1, 2), 'ro'); hold on; 
plot(notr_data(idx2, 1), notr_data(idx2, 2), 'go'); hold on; 
plot(notr_data(idx3, 1), notr_data(idx3, 2), 'bo'); hold off; 
title('특징1 vs. 특징2');
subplot(222); 
plot(notr_data(idx1, 3), notr_data(idx1, 4), 'ro'); hold on; 
plot(notr_data(idx2, 3), notr_data(idx2, 4), 'go'); hold on; 
plot(notr_data(idx3, 3), notr_data(idx3, 4), 'bo'); hold off; 
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
%%
%테스트 데이터를 PCA
ts_scores = nots_data*coeff;

idx1 = find(ts_label == 1);
idx2 = find(ts_label == 2);
idx3 = find(ts_label == 3);

figure;
subplot(221); 
plot(nots_data(idx1, 1), nots_data(idx1, 2), 'ro'); hold on; 
plot(nots_data(idx2, 1), nots_data(idx2, 2), 'go'); hold on; 
plot(nots_data(idx3, 1), nots_data(idx3, 2), 'bo'); hold off; 
title('특징1 vs. 특징2');
subplot(222); 
plot(nots_data(idx1, 3), nots_data(idx1, 4), 'ro'); hold on; 
plot(nots_data(idx2, 3), nots_data(idx2, 4), 'go'); hold on; 
plot(nots_data(idx3, 3), nots_data(idx3, 4), 'bo'); hold off; 
title('특징3 vs. 특징4');
subplot(223); 
plot(ts_scores(idx1, 1), ts_scores(idx1, 2), 'ro'); hold on; 
plot(ts_scores(idx2, 1), ts_scores(idx2, 2), 'go'); hold on; 
plot(ts_scores(idx3, 1), ts_scores(idx3, 2), 'bo'); hold off; 
title('pc1 vs. pc2');
subplot(224); 
plot(ts_scores(idx1, 3), ts_scores(idx1, 4), 'ro'); hold on; 
plot(ts_scores(idx2, 3), ts_scores(idx2, 4), 'go'); hold on; 
plot(ts_scores(idx3, 3), ts_scores(idx3, 4), 'bo'); hold off; 
title('pc3 vs. pc4');
%% PCA특징1, 2로 random forest 생성. 예. 나무 50개
Mdl = TreeBagger(50, tr_score(:,1:2), tr_label, Method="classification");

[est_label] = predict(Mdl, ts_scores(:,1:2));
est_label = cell2mat(est_label);
est_label = str2num(est_label);

figure;
confusionchart(ts_label, est_label);

figure;
subplot(211); bar(est_label); axis tight;
subplot(212); bar(ts_label); axis tight;



