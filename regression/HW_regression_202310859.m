clear; close all; clc;

% 인공지능개론 9주차 과제: 마력이 45 이하인 데이터로 Stepwise 중회귀 모델링

DATA = readmatrix('auto-mpg_ex.csv');
% 1열: mpg (연비), 2열: disp (배기량), 3열: hp (마력), 4열: wt (무게), 5열: accler (가속)

% 마력이 45이하인 데이터만 거르기
idx = find(DATA(:,3) <= 45);

% 종속변수 만들기
y = DATA(idx,1);

% 독립변수 만들기
x = DATA(idx,2:5);

% 그림 그려보기
figure;
subplot(2,2,1); plot(x(:,1), y, 'ro'); xlabel('배기량'); ylabel('연비');
subplot(2,2,2); plot(x(:,2), y, 'ro'); xlabel('마력');   ylabel('연비');
subplot(2,2,3); plot(x(:,3), y, 'ro'); xlabel('무게');   ylabel('연비');
subplot(2,2,4); plot(x(:,4), y, 'ro'); xlabel('가속');   ylabel('연비');

% 상관계수 구해보기
corrcoef(x(:,1), y)
corrcoef(x(:,2), y)
corrcoef(x(:,3), y)
corrcoef(x(:,4), y)

A = x;
A(:,end+1) = ones(length(x), 1);

% 단계적 선택법 (stepwise)
mdl = stepwiselm(A, y);
tmp = mdl.Coefficients
w = tmp.Variables

% 선택된 항: (Intercept), x1(배기량), x2(마력), x3(무게), x1:x3(배기량*무게)
% 추정 결과 만들어 보기 & 그려보기
y_est = w(1,1) + w(2,1)*x(:,1) + w(3,1)*x(:,2) + w(4,1)*x(:,3) + w(5,1)*x(:,1).*x(:,3);

figure;
plot(y, y_est, 'ro'); xlabel('실제연비'); ylabel('추정연비'); xlim([5 35]); ylim([5 35]);
hold on; plot([5 35], [5 35], 'b-'); hold off;

error_1 = mean(abs(y-y_est)) % MAE
error_2 = 100*mean(abs(y-y_est)./y) % %error
error_3 = sqrt(mean((y-y_est).^2)) % RMSE: root mean square error
