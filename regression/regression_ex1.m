clear all; close all; clc;

% 단순회귀분석: 무게로 연비 예측

DATA = readmatrix('auto-mpg_ex.csv');
% 1열: mpg (연비), 2열: disp (배기량), 3열: hp (마력), 4열: wt (무게), 5열: accler (가속)

% 종속(결과)변수 만들기
y = DATA(:,1);

% 독립(설명)변수 만들기
x = DATA(:,4);

A(:,1) = x;
A(:,2) = ones(length(x), 1);

w  = A\y
w2 = ((A'*A)^-1)*A'*y
w3 = regress(y, A)

y_est = w(1)*x + w(2);

figure;
plot(y, y_est, 'ro'); xlabel('실제연비'); ylabel('추정연비'); xlim([5 50]); ylim([5 50]);

error_1 = mean(abs(y-y_est)); % MAE
error_2 = 100*mean(abs(y-y_est)./y); % %error
error_3 = sqrt(mean((y-y_est).^2)); % RMSE: root mean square error