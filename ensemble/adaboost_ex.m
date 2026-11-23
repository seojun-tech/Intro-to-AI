%% adaptive boosting
clear; close all; clc;

load census1994;
Mdl1 = fitcensemble(adultdata,'salary','Method','AdaBoostM1');

ts_data = adulttest(:,1:14);
ts_label = categorical(table2array(adulttest(:,15)));

[est_ts] = predict(Mdl1,ts_data);

figure;
confusionchart(ts_label, est_ts);
