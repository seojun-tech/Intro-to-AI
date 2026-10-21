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
Training_label(:,2) = 1:1:length(Training_label);

ts_id = [1:1:20 51:1:70 101:1:120];
Test_data = meas(ts_id,1:2); % Sepal length/width
Test_label = spcs2num(ts_id,:);

%% 매트랩 내장함수 이용
mdl = fitctree(Training_data, Training_label(:,1));
view(mdl,'Mode','graph');

[matlab_label] = predict(mdl, Test_data);

figure;
confusionchart(Test_label, matlab_label);


% 매트랩 내장함수를 이용한 사후 가지치기 예시
% 최적 leaf 찾기
rng(0); % [수정] cvloss는 무작위로 나눈 10겹 교차검증이라 실행마다 결과가 달라짐 → 재현을 위해 시드 고정
resubcost = resubLoss(mdl,'Subtrees','all'); % 재대입 오차
[cost,secost,ntermnodes,bestlevel] = cvloss(mdl,'Subtrees','all'); % 교차검증 오차
figure; % [수정] 새 창을 열지 않으면 바로 위 confusionchart 창에 겹쳐 그려짐
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
%% 질문 생성
sFeature = [];
sFeature(:,1) = sort(Training_data(:,1));
sFeature(:,2) = sort(Training_data(:,2));

% sepal length/width 각각에 대해 질문을 생성
% 동일 관측값을 제외하며, 인접값의 중간값을 질문으로 만듦
Qdata1 = [];
Qdata2 = [];
cnt1 = 1;
cnt2 = 1;
for k=1:1:length(sFeature)-1
    if sFeature(k+1,1) ~= sFeature(k,1)
        Qdata1(cnt1, 1) = (sFeature(k+1,1) + sFeature(k,1))/2;
        cnt1 = cnt1 + 1;
    end

    if sFeature(k+1,2) ~= sFeature(k,2)
        Qdata2(cnt2, 1) = (sFeature(k+1,2) + sFeature(k,2))/2;
        cnt2 = cnt2 + 1;
    end
end

% 질문을 하나의 변수에 담음. 어떤 변수의 질문인지 모르므로 2열에 인덱스를 넣어둠
allQdata = [];
allQdata = [Qdata1 ones(length(Qdata1), 1); Qdata2 2*ones(length(Qdata2), 1)];

%%
TR_label_dt = {};
TR_label_dt{1} = Training_label;
TR_data_dt = {};
TR_data_dt{1} = Training_data;

Final_Q = [];
EST_LABEL = zeros(length(Training_label), 1);
chk = 1;

% ppp =1;

while(chk == 1)
%     if ppp == 8
%         break;
%     end
    TMP_label_dt = {};
    TMP_data_dt = {};
    cnt = 1;
    [r1, c1] = size(TR_label_dt);
    for j=1:1:c1
        [r2, c2] = size(TR_label_dt{j});
        if r2 < 10
            idx1 = find(TR_label_dt{j}(:,1) == 1);
            idx2 = find(TR_label_dt{j}(:,1) == 2);
            idx3 = find(TR_label_dt{j}(:,1) == 3);

            idx_set = [idx1; idx2; idx3;];
            idx_len = [length(idx1) length(idx2) length(idx3)];
            [mv, mi] = max(idx_len);

            EST_LABEL(TR_label_dt{j}(idx_set,2)) = mi;
        else
            Prev_gini = 1-( (length(find(TR_label_dt{j}(:,1) ==1))^2/length(TR_label_dt{j}(:,1))^2) + ...
                (length(find(TR_label_dt{j}(:,1) ==2))^2/length(TR_label_dt{j}(:,1))^2) + ...
                (length(find(TR_label_dt{j}(:,1) ==3))^2/length(TR_label_dt{j}(:,1))^2) );

            diff_gini_set = [];
            [r3,c3] = size(allQdata);
            for k=1:1:r3
                tmp = allQdata(k,:);
                idx_l = find(TR_data_dt{j}(:,tmp(2)) < tmp(1));
                idx_r = find(TR_data_dt{j}(:,tmp(2)) >= tmp(1));

                tmp_r = TR_label_dt{j}(idx_r,1);
                tmp_gini_r = 1-( (length(find(tmp_r ==1))^2/length(tmp_r)^2) + ...
                    (length(find(tmp_r ==2))^2/length(tmp_r)^2) + ...
                    (length(find(tmp_r ==3))^2/length(tmp_r)^2) );

                tmp_l = TR_label_dt{j}(idx_l,1);
                tmp_gini_l = 1-( (length(find(tmp_l ==1))^2/length(tmp_l)^2) + ...
                    (length(find(tmp_l ==2))^2/length(tmp_l)^2) + ...
                    (length(find(tmp_l ==3))^2/length(tmp_l)^2) );

                diff_gini_set(k, 1) = Prev_gini - (length(idx_l)/(length(idx_l)+length(idx_r)))*tmp_gini_l - (length(idx_r)/(length(idx_l)+length(idx_r)))*tmp_gini_r;
                diff_gini_set(k, 2) = tmp(2);
            end

            %         figure; plot(diff_gini_set(:,1));

            [mv, mi] = max(diff_gini_set(:,1));
            Final_Q = [Final_Q; allQdata(mi,:)];

            idx_l = find(TR_data_dt{j}(:,allQdata(mi,2)) < allQdata(mi,1));
            idx_r = find(TR_data_dt{j}(:,allQdata(mi,2)) >= allQdata(mi,1));

            idx1 = find(TR_label_dt{j}(idx_l,1) == 1);
            idx2 = find(TR_label_dt{j}(idx_l,1) == 2);
            idx3 = find(TR_label_dt{j}(idx_l,1) == 3);

            idx_set = [idx1; idx2; idx3;];
            idx_len = [length(idx1) length(idx2) length(idx3)];

            [sv, si] = sort(idx_len);

            if sv(1)== 0 && sv(2) == 1
                EST_LABEL(TR_label_dt{j}(idx_l(idx_set),2)) = si(3);
            else
                TMP_label_dt{cnt} = TR_label_dt{j}(idx_l,:);
                TMP_data_dt{cnt} = TR_data_dt{j}(idx_l,:);
                cnt = cnt + 1;
            end


            idx1 = find(TR_label_dt{j}(idx_r,1) == 1);
            idx2 = find(TR_label_dt{j}(idx_r,1) == 2);
            idx3 = find(TR_label_dt{j}(idx_r,1) == 3);

            idx_set = [idx1; idx2; idx3;];
            idx_len = [length(idx1) length(idx2) length(idx3)];

            [sv, si] = sort(idx_len);

            if sv(1)== 0 && sv(2) == 1
                EST_LABEL(TR_label_dt{j}(idx_r(idx_set),2)) = si(3);
            else
                TMP_label_dt{cnt} = TR_label_dt{j}(idx_r,:);
                TMP_data_dt{cnt} = TR_data_dt{j}(idx_r,:);
                cnt = cnt + 1;
            end
        end
    end

    [r4, c4] = size(TMP_label_dt);
    if r4 == 0
        chk = 0;
    else
        TR_label_dt = TMP_label_dt;
        TR_data_dt = TMP_data_dt;
    end

%     ppp = ppp+ 1;

end

%%
Final_Q
% [수정] matlab_label은 평가 데이터(60개) 예측값이고 EST_LABEL은 학습 데이터(90개) 분류 결과라
%        길이가 달라 마지막 줄의 빼기에서 오류가 남 → 같은 학습 데이터에 대한 fitctree 예측과 비교
matlab_train_label = predict(mdl, Training_data);
figure;
subplot(311); plot(matlab_train_label);
subplot(312); plot(EST_LABEL);
subplot(313); plot(matlab_train_label(:)-EST_LABEL(:));





