clear; close all; clc;

load fisheriris;

Training_label = [];

for k=1:1:length(species)
    if strcmp(species(k), 'setosa') == 1
        Training_label(k,1) = 0;
    elseif strcmp(species(k), 'versicolor') == 1
        Training_label(k,1) = 1;
    end
end

Training_data = meas(1:100,1:2);
Class = [0; 1];
%%
nb_Model = fitcnb(Training_data,Training_label);

% 모델 평가
matlab_result = predict(nb_Model,Training_data); 
figure;
bar(matlab_result);

%%
[n_d, n_m] = size(Training_data);

prior_set = []; % P(wi), P(wj)
for k=1:1:length(Class)
    prior_set(k,1) = length(find(Training_label == Class(k)))/length(Training_label);
end

mean_set = [];
for k=1:1:n_m % 행: 특징 (sepal length, sepal width)
    for j=1:1:length(Class) % 열: 그룹 (setosa, versicolor)
        idx = find(Training_label == Class(j));
        mean_set(k,j) = mean(Training_data(idx,k));
    end
end
dmean = mean_set(:,2) - mean_set(:,1); % ui - uj (sepal length, sepal width에 대해)
smean = mean_set(:,2) + mean_set(:,1); % ui + uj (sepal length, sepal width에 대해)

% SIGMA = cov(Training_data);
% for k=1:1:length(SIGMA)
%     for j=1:1:length(SIGMA)
%         if k ~= j
%             SIGMA(k, j) = 0;
%         end
%     end
% end

SIGMA = zeros(n_m, n_m);
for k=1:1:length(SIGMA)
    for j=1:1:length(SIGMA)
        if k == j
            SIGMA(k, j) = var((Training_data(:,k)));
        end
    end
end

Gx_grad = transpose(inv(SIGMA)*dmean);
Gx_offset = ( smean/2 - (dmean/(transpose(dmean) * inv(SIGMA) * dmean))*log(prior_set(1,1)/prior_set(2,1)));

my_result = [];
for k=1:1:length(Training_data)
    tmp = Training_data(k,:)';

    if Gx_grad*(tmp-Gx_offset) > 0
        my_result(k,1) = 1;
    else
        my_result(k,1) = 0;
    end
end

%%
figure;
subplot(311);bar(matlab_result); 
subplot(312);bar(my_result); 
subplot(313);plot(matlab_result - my_result); 


%%
idx0 = find(Training_label == 0);
idx1 = find(Training_label == 1);

figure;
plot(Training_data(idx0,1), Training_data(idx0,2), 'bo'); hold on;
plot(Training_data(idx1,1), Training_data(idx1,2), 'go'); hold on;

x_set = [];
x_set(:,1) = 4:0.1:7.5;

Gx = (-Gx_grad(1)/Gx_grad(2))*x_set + (Gx_grad(1)*Gx_offset(1)/Gx_grad(2))+ (Gx_grad(2)*Gx_offset(2)/Gx_grad(2));

plot(x_set, Gx, 'k');