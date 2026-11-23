% run_lab10_save.m
% random_forest_ex.m, adaboost_ex.m를 실행하고 그림·정확도를 저장하는 스크립트
% (원본 실습 코드는 수정하지 않음. 그림은 실행 결과 변수로 다시 그려서 저장)
root = 'C:\Users\vnfmw\Major_2\Intro to AI\Lab10_Ensemble';
cd(root);
if ~exist(fullfile(root,'figures'),'dir'), mkdir(fullfile(root,'figures')); end
figdir = fullfile(root,'figures');

%% 1) Random Forest + Decision Tree (fisheriris)
random_forest_ex;   % 내부에서 clear/close all 수행
root = 'C:\Users\vnfmw\Major_2\Intro to AI\Lab10_Ensemble';
figdir = fullfile(root,'figures');
close all;

rf_acc = mean(est_label == ts_label);
dt_acc = mean(dt_est == ts_label);
rf_cm  = confusionmat(ts_label, est_label);
dt_cm  = confusionmat(ts_label, dt_est);

f = figure('Color','w'); confusionchart(ts_label, est_label); title('Random Forest (50 trees) - fisheriris test');
exportgraphics(f, fullfile(figdir,'rf_confusion.png'), 'Resolution', 150);
f = figure('Color','w'); confusionchart(ts_label, dt_est); title('Decision Tree - fisheriris test');
exportgraphics(f, fullfile(figdir,'dt_confusion.png'), 'Resolution', 150);
f = figure('Color','w');
subplot(211); bar(est_label); axis tight; title('Random Forest 예측');
subplot(212); bar(ts_label);  axis tight; title('정답');
exportgraphics(f, fullfile(figdir,'rf_bar.png'), 'Resolution', 150);
f = figure('Color','w');
subplot(211); bar(dt_est);   axis tight; title('Decision Tree 예측');
subplot(212); bar(ts_label); axis tight; title('정답');
exportgraphics(f, fullfile(figdir,'dt_bar.png'), 'Resolution', 150);

fid = fopen(fullfile(root,'results.txt'),'w');
fprintf(fid, '[fisheriris] train %d / test %d\n', numel(tr_label), numel(ts_label));
fprintf(fid, 'test index: %s\n', mat2str(ts_id));
fprintf(fid, 'RandomForest(50) acc = %.4f (%d/%d)\n', rf_acc, sum(est_label==ts_label), numel(ts_label));
fprintf(fid, 'DecisionTree     acc = %.4f (%d/%d)\n', dt_acc, sum(dt_est==ts_label), numel(ts_label));
fprintf(fid, 'RF confusion (row=true 1..3):\n'); fprintf(fid, '%d %d %d\n', rf_cm');
fprintf(fid, 'DT confusion (row=true 1..3):\n'); fprintf(fid, '%d %d %d\n', dt_cm');
fprintf(fid, 'DT nodes = %d\n', mdl_dt.NumNodes);
fclose(fid);

%% 2) AdaBoost (census1994)
adaboost_ex;   % 내부에서 clear/close all 수행
root = 'C:\Users\vnfmw\Major_2\Intro to AI\Lab10_Ensemble';
figdir = fullfile(root,'figures');
close all;

ab_acc = mean(est_ts == ts_label);
[ab_cm, ab_order] = confusionmat(ts_label, est_ts);
f = figure('Color','w'); confusionchart(ts_label, est_ts); title('AdaBoostM1 - census1994 adulttest');
exportgraphics(f, fullfile(figdir,'adaboost_confusion.png'), 'Resolution', 150);

fid = fopen(fullfile(root,'results.txt'),'a');
fprintf(fid, '\n[census1994] train %d / test %d\n', height(adultdata), numel(ts_label));
fprintf(fid, 'AdaBoostM1 acc = %.4f (%d/%d)\n', ab_acc, sum(est_ts==ts_label), numel(ts_label));
fprintf(fid, 'learners = %d, 1st tree nodes = %d\n', Mdl1.NumTrained, Mdl1.Trained{1}.NumNodes);
fprintf(fid, 'class order: %s\n', strjoin(string(ab_order)', ', '));
fprintf(fid, '%d %d\n', ab_cm');
fclose(fid);

type(fullfile(root,'results.txt'))
