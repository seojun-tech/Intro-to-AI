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


%%
sFeature = [];
sFeature(:,1) = sort(Training_data(:,1));
sFeature(:,2) = sort(Training_data(:,2));

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

while(chk == 1)
    TMP_label_dt = {};
    TMP_data_dt = {};
    cnt = 1;
    for j=1:1:length(TR_label_dt)
        Prev_gini = 1-( (length(find(TR_label_dt{j}(:,1) ==1))^2/length(TR_label_dt{j}(:,1))^2) + ...
            (length(find(TR_label_dt{j}(:,1) ==2))^2/length(TR_label_dt{j}(:,1))^2) + ...
            (length(find(TR_label_dt{j}(:,1) ==3))^2/length(TR_label_dt{j}(:,1))^2) );

        diff_gini_set = [];
        for k=1:1:length(allQdata)
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
        Final_Q = [Final_Q; allQdata(mi,:)]

        idx_l = find(TR_data_dt{j}(:,allQdata(mi,2)) < allQdata(mi,1));
        idx_r = find(TR_data_dt{j}(:,allQdata(mi,2)) >= allQdata(mi,1));

        idx1 = find(TR_label_dt{j}(idx_l,1) == 1);
        idx2 = find(TR_label_dt{j}(idx_l,1) == 2);
        idx3 = find(TR_label_dt{j}(idx_l,1) == 3);

        l_len1 = [ length(idx1) length(idx2) length(idx3)];
        l_len1 = sort(l_len1);

        if l_len1(2) == 0
            EST_LABEL(TR_label_dt{j}(idx_l(idx1),2)) = 1;
            EST_LABEL(TR_label_dt{j}(idx_l(idx2),2)) = 2;
            EST_LABEL(TR_label_dt{j}(idx_l(idx3),2)) = 3;
        else
            TMP_label_dt{cnt} = TR_label_dt{j}(idx_l,:);
            TMP_data_dt{cnt} = TR_data_dt{j}(idx_l,:);
            cnt = cnt + 1;
        end

        idx1 = find(TR_label_dt{j}(idx_r,1) == 1);
        idx2 = find(TR_label_dt{j}(idx_r,1) == 2);
        idx3 = find(TR_label_dt{j}(idx_r,1) == 3);

        l_len2 = [ length(idx1) length(idx2) length(idx3)];
        l_len2 = sort(l_len2);
        if l_len2(2) == 0
            EST_LABEL(TR_label_dt{j}(idx_r(idx1),2)) = 1;
            EST_LABEL(TR_label_dt{j}(idx_r(idx2),2)) = 2;
            EST_LABEL(TR_label_dt{j}(idx_r(idx3),2)) = 3;
        else
            TMP_label_dt{cnt} = TR_label_dt{j}(idx_r,:);
            TMP_data_dt{cnt} = TR_data_dt{j}(idx_r,:);
            cnt = cnt + 1;
        end
    end

    if (l_len2(2) == 0 && l_len1(2) == 0) || (length(idx_l) < 3 && (length(idx_r) < 3))
         idx1 = find(TR_label_dt{j}(idx_l,1) == 1);
         idx2 = find(TR_label_dt{j}(idx_l,1) == 2);
         idx3 = find(TR_label_dt{j}(idx_l,1) == 3);
         EST_LABEL(TR_label_dt{j}(idx_l(idx1),2)) = 1;
         EST_LABEL(TR_label_dt{j}(idx_l(idx2),2)) = 2;
         EST_LABEL(TR_label_dt{j}(idx_l(idx3),2)) = 3;

         idx1 = find(TR_label_dt{j}(idx_r,1) == 1);
         idx2 = find(TR_label_dt{j}(idx_r,1) == 2);
         idx3 = find(TR_label_dt{j}(idx_r,1) == 3);
         EST_LABEL(TR_label_dt{j}(idx_r(idx1),2)) = 1;
         EST_LABEL(TR_label_dt{j}(idx_r(idx2),2)) = 2;
         EST_LABEL(TR_label_dt{j}(idx_r(idx3),2)) = 3;

        chk = 0;
    else
        TR_label_dt = TMP_label_dt;
        TR_data_dt = TMP_data_dt;      
    end

%         if length(idx_l) >= 5 || length(idx_r) >= 5
%             Final_Q = [Final_Q; allQdata(mi,:)]
%         end
% 
%         if length(idx_l) >= 5
%             TMP_label_dt{cnt} = TR_label_dt{j}(idx_l,:);
%             TMP_data_dt{cnt} = TR_data_dt{j}(idx_l,:);
%             cnt = cnt + 1;
%         else
%             idx1 = find(TR_label_dt{j}(idx_l,1) == 1);
%             idx2 = find(TR_label_dt{j}(idx_l,1) == 2);
%             idx3 = find(TR_label_dt{j}(idx_l,1) == 3);
%             EST_LABEL(TR_label_dt{j}(idx_l(idx1),2)) = 1;
%             EST_LABEL(TR_label_dt{j}(idx_l(idx2),2)) = 2;
%             EST_LABEL(TR_label_dt{j}(idx_l(idx3),2)) = 3;
%         end
% 
%         if length(idx_r) >= 5
%             TMP_label_dt{cnt} = TR_label_dt{j}(idx_r,:);
%             TMP_data_dt{cnt} = TR_data_dt{j}(idx_r,:);            
%             cnt = cnt + 1;
%         else
%             idx1 = find(TR_label_dt{j}(idx_r,1) == 1);
%             idx2 = find(TR_label_dt{j}(idx_r,1) == 2);
%             idx3 = find(TR_label_dt{j}(idx_r,1) == 3);
%             EST_LABEL(TR_label_dt{j}(idx_r(idx1),2)) = 1;
%             EST_LABEL(TR_label_dt{j}(idx_r(idx2),2)) = 2;
%             EST_LABEL(TR_label_dt{j}(idx_r(idx3),2)) = 3;
%         end
%     end

    

%     if length(idx_l) < 5 && length(idx_r) < 5
%         idx1 = find(TR_label_dt{j}(idx_r,1) == 1);
%         idx2 = find(TR_label_dt{j}(idx_r,1) == 2);
%         idx3 = find(TR_label_dt{j}(idx_r,1) == 3);
%         EST_LABEL(TR_label_dt{j}(idx_r(idx1),2)) = 1;
%         EST_LABEL(TR_label_dt{j}(idx_r(idx2),2)) = 2;
%         EST_LABEL(TR_label_dt{j}(idx_r(idx3),2)) = 3;
% 
%         idx1 = find(TR_label_dt{j}(idx_l,1) == 1);
%         idx2 = find(TR_label_dt{j}(idx_l,1) == 2);
%         idx3 = find(TR_label_dt{j}(idx_l,1) == 3);
%         EST_LABEL(TR_label_dt{j}(idx_l(idx1),2)) = 1;
%         EST_LABEL(TR_label_dt{j}(idx_l(idx2),2)) = 2;
%         EST_LABEL(TR_label_dt{j}(idx_l(idx3),2)) = 3;
% 
%         chk = 0;
%     else
%         TR_label_dt = TMP_label_dt;
%         TR_data_dt = TMP_data_dt;
%         
%     end
end

%%
figure;
subplot(211); plot(matlab_label);
subplot(212); plot(EST_LABEL);





