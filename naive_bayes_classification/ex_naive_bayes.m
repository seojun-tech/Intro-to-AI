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

Training_data = meas(1:100,1); % sepal length만
%%
nb_Model = fitcnb(Training_data,Training_label);

% 모델 평가
matlab_result = predict(nb_Model,Training_data); 
figure;
subplot(211); bar(matlab_result);
subplot(212); bar(Training_label);

%%
% 사전 확률 계산
PA1 = length(find(Training_label == 0))/length(Training_label); % setosa의 사전확률 P(A=setosa)
PA2 = length(find(Training_label == 1))/length(Training_label); % versicolor의 사전확률 P(A=versicolor)

% 조건부 확률 그래프 도출
x = 0:0.1:10;

m1 = mean(Training_data(1:50,1)); % setosa sepal length의 평균
s1 = std(Training_data(1:50,1));  % setosa sepal length의 표준편차
m2 = mean(Training_data(51:100,1)); % versicolor의 sepal length의 평균
s2 = std(Training_data(51:100,1)); % versicolor의 sepal length의 표준편차

% 정규 분포 그래프 만들기
PBA1 = ( 1/(s1*sqrt(2*pi)) ) * exp( (-(x-m1).^2)/(2*(s1.^2)) );
PBA1 = PBA1/sum(PBA1); % setosa sepal length의 조건부 확률 그래프 사전확률 P(B|A=setosa)
PBA2 = ( 1/(s2*sqrt(2*pi)) ) * exp( (-(x-m2).^2)/(2*(s2.^2)) );
PBA2 = PBA2/sum(PBA2);  % versicolor sepal length의 조건부 확률 그래프 사전확률 P(B|A=versicolor)

figure;
plot(x, PBA1); hold on; plot(x, PBA2); legend([{'setota'}; {'versicolor'}])

figure;
plot(x, PBA1, 'bo--'); hold on; plot(x, PBA2, 'ro--'); legend([{'setota'}; {'versicolor'}])

my_result = [];
for k=1:1:length(Training_data)
    x = Training_data(k);
    tPBA1 = ( 1/(s1*sqrt(2*pi)) ) * exp( (-(x-m1).^2)/(2*(s1.^2)) );
    tPBA2 = ( 1/(s2*sqrt(2*pi)) ) * exp( (-(x-m2).^2)/(2*(s2.^2)) );
   
    if PA1*tPBA1 > PA2*tPBA2 % P(A=setosa)*P(B|A=setosa) 와 P(A=versicolor)*P(B|A=versicolor)를 비교하여 큰 것을 결정
        my_result(k, 1) = 0;
    else
        my_result(k, 1) = 1;
    end
end

figure;
subplot(311);bar(matlab_result); 
subplot(312);bar(my_result); 
subplot(313);plot(matlab_result - my_result); 

%%
% 사전 확률 계산
PA1 = length(find(Training_label == 0))/length(Training_label); % setosa의 사전확률 P(A=setosa)
PA2 = length(find(Training_label == 1))/length(Training_label); % versicolor의 사전확률 P(A=versicolor)


m1 = mean(Training_data(1:50,1)); % setosa sepal length의 평균
s1 = std(Training_data(1:50,1));  % setosa sepal length의 표준편차
m2 = mean(Training_data(51:100,1)); % versicolor의 sepal length의 평균
s2 = std(Training_data(51:100,1)); % versicolor의 sepal length의 표준편차

my_result = [];
for k=1:1:length(Training_data)
    x = Training_data(k);    
    PBA1 = ( 1/(s1*sqrt(2*pi)) ) * exp( (-(x-m1).^2)/(2*(s1.^2)) );   
    PBA2 = ( 1/(s2*sqrt(2*pi)) ) * exp( (-(x-m2).^2)/(2*(s2.^2)) );

    if PA1*PBA1 > PA2*PBA2 % P(A=setosa)*P(B|A=setosa) 와 P(A=versicolor)*P(B|A=versicolor)를 비교하여 큰 것을 결정
        my_result(k, 1) = 0;
    else
        my_result(k, 1) = 1;
    end
end

figure;
subplot(311);bar(matlab_result); 
subplot(312);bar(my_result); 
subplot(313);plot(matlab_result - my_result); 
%% 정확도 계산해보기

% Training_label과 my_result가 같은 것 (제대로 찾은 것)이 몇개 인가?
% Training_label == my_result : 같으면 1, 다르면 0인 벡터
% find : 값이 1(같은 것)인 위치(인덱스)들을 찾아줌 -> 그 개수가 맞힌 개수
num_correct = length(find(Training_label == my_result));

% 전체 평가 데이터는 몇개 인가?
num_total = length(Training_label);

% 전체 평가 데이터 개수에서 제대로 찾은 것의 비율이 정확도가 될 것
accuracy = num_correct / num_total * 100;

fprintf('제대로 찾은 개수 : %d개\n', num_correct);
fprintf('전체 평가 데이터 : %d개\n', num_total);
fprintf('정확도 : %.1f%%\n', accuracy);

% (참고) MATLAB fitcnb 결과(matlab_result)의 정확도도 같은 방법으로 비교
matlab_accuracy = length(find(Training_label == matlab_result)) / num_total * 100;
fprintf('MATLAB fitcnb 정확도 : %.1f%%\n', matlab_accuracy);
