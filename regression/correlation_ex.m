clear; close all; clc;


DATA = readmatrix('auto-mpg_ex.csv');
% 1열: mpg (연비), 2열: disp (배기량), 3열: hp (마력), 4열: wt (무게), 5열: accler (가속)

% 종속(결과)변수 만들기
y = DATA(:,1);

% 독립(설명)변수 만들기
x = DATA(:,2:5);

% 상관관계 확인
corrcoef(x(:,1), y)
corrcoef(x(:,2), y)
corrcoef(x(:,3), y)
corrcoef(x(:,4), y)

% 그래프 그래보기
figure;
subplot(2,2,1); plot(x(:,1), y, 'ro'); % x축: 배기량, y축: 연비
subplot(2,2,2); plot(x(:,2), y, 'ro'); % x축: 마력, y축: 연비
subplot(2,2,3); plot(x(:,3), y, 'ro'); % x축: 무게, y축: 연비
subplot(2,2,4); plot(x(:,4), y, 'ro'); % x축: 가속, y축: 연비
