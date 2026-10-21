clear; close all; clc;

load fisheriris;

% 각 종을 숫자로 표현 1: setosa, 2: versicolor, 3: virginica
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

% 학습 데이터와 평가 데이터 나누기
% 먼저 Sepal length/width 정보를 이용하여 두개의 그룹을 나눠봅시다 versicolor vs. virginica
% 오늘은 편의상 아래와 같이 나누겠음
% 1~50: setosa, 51~100: versicolor, 101~150: virginica
% 학습데이터. 21~50: setosa, 71~100: versicolor, 121~150: virginica 총 90개
% 평가데이터. 1~20: setosa, 51~70: versicolor, 101~120: virginica 총 60개
tr_id = [21:1:50 71:1:100 121:1:150];
Training_data = meas(tr_id,1:2); % Sepal length/width
Training_label = spcs2num(tr_id,:);

ts_id = [1:1:20 51:1:70 101:1:120];
Test_data = meas(ts_id,1:2); % Sepal length/width
Test_label = spcs2num(ts_id,:);

%% 매트랩 내장함수 이용
mdl = fitctree(Training_data, Training_label);
view(mdl,'Mode','graph');

[matlab_label] = predict(mdl, Test_data);

figure;
subplot(211); bar(Test_label); axis tight;
subplot(212); bar(matlab_label); axis tight;

figure;
confusionchart(Test_label, matlab_label);


% 매트랩 내장함수를 이용한 사후 가지치기 예시
% 최적 leaf 찾기
resubcost = resubLoss(mdl,'Subtrees','all'); % 재대입 오차
[cost,secost,ntermnodes,bestlevel] = cvloss(mdl,'Subtrees','all'); % 교차검증 오차
plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
figure(gcf);
xlabel('Number of terminal nodes');
ylabel('Cost (misclassification error)')
legend('Cross-validation','Resubstitution')

% 최적점 찾기
[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
plot([0 20], [cutoff cutoff], 'k:')
plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')
hold off

pt = prune(mdl,'Level',bestlevel);
view(pt,'Mode','graph')

matlab_label2 = predict(pt, Test_data);
figure;
confusionchart(Test_label, matlab_label2)
