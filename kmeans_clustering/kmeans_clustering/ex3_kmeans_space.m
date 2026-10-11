clear all; close all; clc;

load fisheriris;

% sepal length, width만 이용해보자
setdata = [];
setdata = meas(:,1:2); %3:4 % 1:4

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

% 매트랩 내부함수. 출력: mat_label: 그룹 할당 결과, mat_CP: 각 그룹의 중심 좌표
% [mat_label, mat_CP] = kmeans(setdata, 3);
[mat_label, mat_CP] = kmeans(setdata, 3, 'Start', setdata(1:3,:));

figure;
subplot(211); bar(mat_label); axis tight; title('추정: kmeans');
subplot(212); bar(spcs2num); axis tight; title('정답');
%% my_kmeans
grp = 3; % k = 3;
class = [1 2 3];

% [0단계: 초기 객체 선정]
% 편의상 1~3번째 데이터를 초기 값으로 하겠음 (거리들을 모두 계산하고 가장 거리가 먼 3점으로 하면 아마도 아래 while문의 동작 횟수가 줄어 들 것)
% (k가 3이니, 3개의 중심좌표가 나와야 하고, sepal length, width 두 정보만 사용하니, 중심좌표는 2차원)
my_CP = setdata(1:grp,:); 

% [1단계: 객체 군집 배정]
my_label = [];
for kk=1:1:length(setdata)
    tmp_set = setdata(kk,:); % kk번째 데이터 가져오기

    % 각 중심좌표와 데이터 간의 유클리디안 거리를 계산
    udist = zeros(grp, 1);
    for jj=1:1:grp
        udist(jj,1) = norm(tmp_set - my_CP(jj, :));
    end

    % 현재 데이터와 중심좌표들의 거리를 오름차순으로 정렬
    [sv, si] = sort(udist, 'ascend');

    % 제일 앞에 있는 것이 거리가 제일 짧은 것. 그것의 그룹 ID로 현재 데이터를 할당하라 
    my_label(kk,1) = class(si(1));
end

Final_label = [];
chk = 1;
loop = 0;
% [2, 3단계, 1단계 반복: 군집 중심좌표 산출 & 수렴 조건 점검 / 객체 군집 배정
% 군집 중심좌표 산출 /객체 군집 배정 를 반복해야 함. 언제까지? 수렴 조건에 수렴할 때까지. 
% 수렴조건의 만족은 for문으로 확인이 어려우므로 while문으로
while(chk)
    loop = loop + 1;
    loop
    for kk=1:1:grp
        idx = find(my_label == class(kk)); % kk 그룹에 할당된 데이터 가져오기
        tmp_data = setdata(idx,:);

        [r1, c1] = size(tmp_data);

        % kk 그룹에 할당된 데이터를 이용하여 중심 좌표를 계산
        my_CP(kk,:) = mean(tmp_data);       
    end

    % 새로운 중심좌표를 이용하여, 데이터를 재할당
    my_label_new = [];
    for kk=1:1:length(setdata)
        tmp_set = setdata(kk,:);

        udist = zeros(grp, 1);
        for jj=1:1:grp
            udist(jj,1) = norm(tmp_set - my_CP(jj, :));
        end

        [sv, si] = sort(udist, 'ascend');

        my_label_new(kk,1) = class(si(1));
    end

    % 1단계의 결과와 2단계의 결과가 같은지 평가
    idx = find(my_label_new ~= my_label);

    % 이전 할당 결과와 지금 할당 결과가 같으면 while문 종료
    % 그렇지 않으면, 위 과정을 반복
    if length(idx) == 0
        chk = 1;
        Final_label = my_label_new;
        break;
    else
        my_label = my_label_new;
    end
end

%
figure;
subplot(311); bar(mat_label); axis tight;
subplot(312); bar(Final_label); axis tight;
subplot(313); bar(mat_label-Final_label); axis tight;

my_CP
mat_CP

%% 공간 분리
tmp1 = min(meas(:,1)):0.01:max(meas(:,1));
tmp2 = linspace(min(meas(:,2)), max(meas(:,2)), length(tmp1)); % linspnace(a, b, c) a부터 b까지 c개가 되도록 데이터를 생성하는 함수

[out1,out2] = meshgrid(tmp1,tmp2);

Test_data = [];
Test_data = [out1(:) out2(:)];

grp = 3; % k = 3;
class = [1 2 3];

[r, c] = size(Test_data);

% [1단계: 객체 군집 배정
Final_label = [];
for kk=1:1:r
    tmp_set = Test_data(kk,:); % kk번째 데이터 가져오기

    % 각 중심좌표와 데이터 간의 유클리디안 거리를 계산
    udist = zeros(grp, 1);
    for jj=1:1:grp
        udist(jj,1) = norm(tmp_set - my_CP(jj, :));
    end

    % 현재 데이터와 중심좌표들의 거리를 오름차순으로 정렬
    [sv, si] = sort(udist, 'ascend');

    % 제일 앞에 있는 것이 거리가 제일 짧은 것. 그것의 그룹 ID로 현재 데이터를 할당하라 
    Final_label(kk,1) = class(si(1));
end


idx1 = find(Final_label == 1);
idx2 = find(Final_label == 2);
idx3 = find(Final_label == 3);

figure;
subplot(211); plot(Test_data(idx1,1), Test_data(idx1,2), 'r.'); hold on;
subplot(211); plot(Test_data(idx2,1), Test_data(idx2,2), 'g.'); hold on;
subplot(211); plot(Test_data(idx3,1), Test_data(idx3,2), 'b.'); hold off;


idx1 = find(spcs2num == 1);
idx2 = find(spcs2num == 2);
idx3 = find(spcs2num == 3);


subplot(212); plot(meas(idx1,1), meas(idx1,2), 'b.'); hold on;
subplot(212); plot(meas(idx2,1), meas(idx2,2), 'g.'); hold on;
subplot(212); plot(meas(idx3,1), meas(idx3,2), 'r.'); hold off;

    




