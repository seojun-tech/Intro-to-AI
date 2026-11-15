clear; close all; clc;


DATA = readmatrix('auto-mpg_ex.csv');
% 1열: mpg (연비), 2열: disp (배기량), 3열: hp (마력), 4열: wt (무게), 5열: accler (가속)

% 종속변수 만들기
y = DATA(:,1);

% 독립변수 만들기
x = DATA(:,2:5);

A = x;
A(:,end+1) = ones(length(x), 1);

% 단계적 선택법 (stepwise)

mdl = stepwiselm(A, y);
tmp=mdl.Coefficients
w=tmp.Variables

% 추정 결과 만들어 보기 & 그려보기
y_est = w(1,1) + w(2,1)*x(:,1) + w(3,1)*x(:,3) + w(4,1)*x(:,4) + w(5,1)*x(:,1).*x(:,3);


figure;
plot(y, y_est, 'ro'); xlabel('실제연비'); ylabel('추정연비');  xlim([5 50]); ylim([5 50]);

error_1 = mean(abs(y-y_est)); % MAE
error_2 = 100*mean(abs(y-y_est)./y); % %error
error_3 = sqrt(mean((y-y_est).^2)); % RMSE: root mean square error