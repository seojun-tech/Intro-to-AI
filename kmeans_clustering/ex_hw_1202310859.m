clear all; close all; clc;

load fisheriris.mat;

% sepal length, width, petal length, width 모두 이용
setdata = [];
setdata = meas(:,1:4);

% 매트랩 내부함수. 출력: mat_label: 그룹 할당 결과, mat_CP: 각 그룹의 중심 좌표
[mat_label, mat_CP] = kmeans(setdata, 3, 'Start', setdata(1:3,:));

%%
grp = 3; % k = 3;
class = [1 2 3];

% 수업시간에 구현한 부분을 my_kmeans 함수로 대체
[Final_label, my_CP] = my_kmeans_202310859(setdata, grp, class);

figure;
subplot(311); plot(mat_label); axis tight; title('매트랩 내장함수 kmeans');
subplot(312); plot(Final_label); axis tight; title('my\_kmeans\_202310859');
subplot(313); plot(mat_label-Final_label); axis tight; title('차이 (mat\_label - Final\_label)');

my_CP
mat_CP

% 결과 비교
fprintf('할당 결과가 다른 데이터 수: %d\n', sum(mat_label ~= Final_label));
fprintf('중심좌표 최대 오차: %e\n', max(abs(my_CP(:) - mat_CP(:))));
