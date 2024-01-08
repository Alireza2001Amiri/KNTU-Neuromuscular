clc
clear all
close all
%% *Load and display data*

load('D:\University\Khaje\Semester 1\کنترل سیستم های عصبی عضلانی\Assignment 3\data.mat')

Raw_Horizontal = data.A;
Raw_Vertical = data.B;

figure(1)
plot(Raw_Horizontal , Raw_Vertical)
title('Vetical vs Horizontal')

%% Detrend data and plot Signals
% Question 1

Detrended_Raw_Horizontal = detrend(Raw_Horizontal);
Detrended_Raw_Vertical = detrend(Raw_Vertical);

figure(2)
title('Raw signals vs Detrended Signals')
hold on
plot(1:length(Raw_Horizontal) , Raw_Horizontal);
plot(1:length(Raw_Vertical) , Raw_Vertical);
plot(1:length(Detrended_Raw_Horizontal) , Detrended_Raw_Horizontal);
plot(1:length(Detrended_Raw_Vertical), Detrended_Raw_Vertical);
legend('Raw Horizontal', 'Raw Vertical' , 'Detrended Raw Horizontal' ,'Detrended Raw Vertical' , 'Location','best')




