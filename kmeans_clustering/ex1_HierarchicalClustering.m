%% 계층적 군집 1
clear; close all; clc;

X = [1 2;2.5 4.5;2 2;4 1.5;4 2.5];

figure;
plot(X(:,1), X(:,2), 'ko'); xlim([0 5]); ylim([1 5]); grid on;

Y = pdist(X)
squareform(Y)

% 단일 연결법, 유클리디안 거리, 계층적 군집
Z = linkage(Y)

% 덴드로그램 그리기
figure;
dendrogram(Z)

% 그루핑
T = cluster(Z,'maxclust',2)
T = cluster(Z,'maxclust',3)

%%
clear; close all; clc; 

load fisheriris;

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

data = meas(:, 3:4); % petal length, width만 가져와라
label = spcs2num;

idx1 = find(label == 1); % setosa만 찾기
idx2 = find(label == 2); % versicolor만 찾기 
idx3 = find(label == 3); % virginica만 찾기
figure;
plot(meas(idx1,3), meas(idx1,4), 'r.'); hold on; % setosa는 빨간색으로 
plot(meas(idx2,3), meas(idx2,4), 'go'); hold on; % versicolor는 녹색으로 
plot(meas(idx3,3), meas(idx3,4), 'bx'); hold on; % virginica는 파란색으로 
xlabel('Petal length');
ylabel('Petal width');

Y = pdist(data)
squareform(Y);

% 단일 연결법, 유클리디안 거리, 계층적 군집
Z = linkage(Y);

% 덴드로그램 그리기
figure;
dendrogram(Z);

% 그루핑
T = cluster(Z,'maxclust',3);

figure;
subplot(211); bar(label); axis tight;
subplot(212); bar(T); axis tight;

