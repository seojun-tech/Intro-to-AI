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

[matlab_label] = predict(mdl, Training_data);

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
% 이 셀형 변수에 그룹핑된 데이터를 저장하여 처리함
TR_label_dt = {};
TR_label_dt{1} = Training_label;
TR_data_dt = {};
TR_data_dt{1} = Training_data; % XT

Final_Q = [];
EST_LABEL = zeros(length(Training_label), 1);
chk = 1;

% 멈춤 조건에 도달할 때까지 루프를 돔 
while(chk == 1)
    TMP_label_dt = {};
    TMP_data_dt = {};
    cnt = 1;
    [r1, c1] = size(TR_label_dt);
    % 처리해야할 데이터의 수 만큼 루프를 돔
    for j=1:1:c1
        [r2, c2] = size(TR_label_dt{j});

        if r2 < 10 % 노드의 데이터가 10개보다 적다면, 최대 빈도를 보이는 그룹으로 할당 
            idx1 = find(TR_label_dt{j}(:,1) == 1);
            idx2 = find(TR_label_dt{j}(:,1) == 2);
            idx3 = find(TR_label_dt{j}(:,1) == 3);

            idx_set = [idx1; idx2; idx3;];
            idx_len = [length(idx1) length(idx2) length(idx3)];
            [mv, mi] = max(idx_len);

            EST_LABEL(TR_label_dt{j}(idx_set,2)) = mi;
        else % 이 부분에 메인
            % 분석할 데이터의 불순도를 계산 (지니계수로)
            Prev_gini = 1-( (length(find(TR_label_dt{j}(:,1) ==1))^2/length(TR_label_dt{j}(:,1))^2) + ...
                (length(find(TR_label_dt{j}(:,1) ==2))^2/length(TR_label_dt{j}(:,1))^2) + ...
                (length(find(TR_label_dt{j}(:,1) ==3))^2/length(TR_label_dt{j}(:,1))^2) );

            % 생성한 질문들에 대해 L, R로 나누었을 때의 불순도 감소량을 계산
            % diff_gini_set의 1열: 불순도 감소량, 2열: 어떤 질문인지 (sepal length width)
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

            % figure; plot(diff_gini_set(:,1));

            % 불순도 감소량이 가장 큰 질문을 찾음
            [mv, mi] = max(diff_gini_set(:,1));
            Final_Q = [Final_Q; allQdata(mi,:)];

            % 불순도 감소량이 가장 큰 질문에 대해 그룹핑을 해봄
            idx_l = find(TR_data_dt{j}(:,allQdata(mi,2)) < allQdata(mi,1));
            idx_r = find(TR_data_dt{j}(:,allQdata(mi,2)) >= allQdata(mi,1));

            idx1 = find(TR_label_dt{j}(idx_l,1) == 1);
            idx2 = find(TR_label_dt{j}(idx_l,1) == 2);
            idx3 = find(TR_label_dt{j}(idx_l,1) == 3);

            idx_set = [idx1; idx2; idx3;];
            idx_len = [length(idx1) length(idx2) length(idx3)];

            [sv, si] = sort(idx_len);
            
            % sv의 가장 작은 값이 0이라는 것은, 어떤 그룹으로는 하나도 분류된 것이 없다는 의미 
            % 즉, 2개의 그룹에 대한 데이터만 있다
            % sv의 중간 값이 1이라는 것은 1개의 데이터가 어떤 그룹으로 분류가 되고, 나머지는 다른 그룹으로 분류됐다는 의미
            % 어떤 그룹의 데이터가 1개 다른 그룹의 데이터가 n개인 상황으로, 과적합방지를 위해 (사전가지치기로) 
            % 그냥 많은 부류로 할당하도록 구현
            if sv(1)== 0 && sv(2) == 1
                EST_LABEL(TR_label_dt{j}(idx_l(idx_set),2)) = si(3); 
            else % 위 상황이 만족되지 않았다면, 데이터를 계속 분할해 나가야 함 
                TMP_label_dt{cnt} = TR_label_dt{j}(idx_l,:);
                TMP_data_dt{cnt} = TR_data_dt{j}(idx_l,:);
                cnt = cnt + 1;
            end

            % 위와 동일한 과정을 R에도 적용
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

    % 더이상 분석할 데이터가 없다 (= 분할된 데이터가 없다면), while문 종료
    % 아니라면, 계속 진행
    [r4, c4] = size(TMP_label_dt);
    if r4 == 0
        chk = 0;
    else
        TR_label_dt = TMP_label_dt;
        TR_data_dt = TMP_data_dt;
    end
end


%%
Final_Q
figure;
subplot(311); bar(matlab_label);
subplot(312); bar(EST_LABEL);
subplot(313); bar(matlab_label(:)-EST_LABEL(:));





