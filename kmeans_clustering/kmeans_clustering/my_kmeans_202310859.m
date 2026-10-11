function [Final_label, my_CP] = my_kmeans_202310859(setdata, grp, class)
% my_kmeans_202310859 : k-means 군집 직접 구현
%   [Final_label, my_CP] = my_kmeans_202310859(setdata, grp, class)
%
% 입력
%   setdata : 입력 데이터 (N x D, 행 = 객체, 열 = 특징)
%   grp     : 그룹 수 (k)
%   class   : 클래스 번호 배열 (예: [1 2 3])
% 출력
%   Final_label : 각 객체에 할당된 그룹 (N x 1)
%   my_CP       : 각 그룹의 중심 좌표 (grp x D)

[N, D] = size(setdata);

% [0단계: 초기 객체 선정]
% 1 ~ grp번째 데이터를 초기 중심좌표로 사용 (매트랩 kmeans의 'Start', setdata(1:grp,:)와 동일)
my_CP = setdata(1:grp, :);

% [1단계: 객체 군집 배정]
my_label = zeros(N, 1);
for kk = 1:1:N
    tmp_set = setdata(kk, :);           % kk번째 데이터

    % 각 중심좌표와 데이터 간의 유클리디안 거리 계산
    udist = zeros(grp, 1);
    for jj = 1:1:grp
        udist(jj, 1) = norm(tmp_set - my_CP(jj, :));
    end

    % 거리가 가장 짧은 그룹으로 할당
    [sv, si] = sort(udist, 'ascend');
    my_label(kk, 1) = class(si(1));
end

% [2, 3단계, 1단계 반복: 중심좌표 산출 -> 재할당 -> 수렴 조건 점검]
Final_label = [];
chk = 1;
while (chk)
    % [2단계: 군집 중심좌표 산출]
    for kk = 1:1:grp
        idx = find(my_label == class(kk));  % kk 그룹에 할당된 데이터
        if ~isempty(idx)                    % 빈 군집이면 이전 중심좌표 유지 (NaN 방지)
            tmp_data = setdata(idx, :);
            my_CP(kk, :) = mean(tmp_data, 1);
        end
    end

    % 새로운 중심좌표로 데이터 재할당
    my_label_new = zeros(N, 1);
    for kk = 1:1:N
        tmp_set = setdata(kk, :);

        udist = zeros(grp, 1);
        for jj = 1:1:grp
            udist(jj, 1) = norm(tmp_set - my_CP(jj, :));
        end

        [sv, si] = sort(udist, 'ascend');
        my_label_new(kk, 1) = class(si(1));
    end

    % [3단계: 수렴 조건 점검] 이전 할당 결과와 같으면 종료, 아니면 반복
    idx = find(my_label_new ~= my_label);
    if isempty(idx)
        chk = 0;
        Final_label = my_label_new;
    else
        my_label = my_label_new;
    end
end

end
