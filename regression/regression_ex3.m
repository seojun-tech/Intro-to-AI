clear; close all; clc;

DATA = readmatrix('auto-mpg_ex.csv');
% 1열: mpg (연비), 2열: disp (배기량), 3열: hp (마력), 4열: wt (무게), 5열: accler (가속)

% 마력이 45이하인 데이터만 거르기
idx = find(DATA(:,3) <= 45);

% 종속변수 만들기
y = DATA(idx,1);

% 독립변수 만들기
x = DATA(idx,2:5);

% 그림 여러개 그려보기
figure;
subplot(2,2,1); plot(x(:,1), y, 'ro'); % x축: 배기량, y축: 연비
subplot(2,2,2); plot(x(:,2), y, 'ro'); % x축: 마력, y축: 연비
subplot(2,2,3); plot(x(:,3), y, 'ro'); % x축: 무게, y축: 연비
subplot(2,2,4); plot(x(:,4), y, 'ro'); % x축: 가속, y축: 연비

% 상관계수 구해보기 
corrcoef(x(:,1), y)
corrcoef(x(:,2), y)
corrcoef(x(:,3), y)
corrcoef(x(:,4), y)

A = x;
A(:,end+1) = ones(length(x), 1);

w  = A\y
w2 = ((A'*A)^-1)*A'*y
w3 = regress(y, A)

% 추정 결과 만들어 보기 & 그려보기
y_est = w(1)*x(:,1) + w(2)*x(:,2) + w(3)*x(:,3) + w(4)*x(:,4) + w(5);

figure;
plot(y, y_est, 'ro'); xlabel('실제연비'); ylabel('추정연비'); xlim([5 50]); ylim([5 50]);

error_1 = mean(abs(y-y_est)); % MAE
error_2 = 100*mean(abs(y-y_est)./y); % %error
error_3 = sqrt(mean((y-y_est).^2)); % RMSE: root mean square error
