% An example to get the BCI competition IV datasets 2a, 2b is the same
% Data from: http://www.bbci.de/competition/iv/
% using open-source toolbox Biosig on MATLAB: http://biosig.sourceforge.net/
% Just an example, you should change as you need.
clear all

subject_index =9; % 1-9

%% T data
session_type = 'T'; % T and E

load(strcat('D:\Projects\DataSet\BCICIV_2a_mat\A0',num2str(subject_index),session_type))
label_1 = [];
k = 0;
data_1 = zeros(1000,22,288);
for i = 4:9
% for i = 2:7     %when subject_index=4
    label_1 = [label_1;data{i}.y];
    Pos = data{i}.trial;
    for j = 1:numel(Pos)
        k = k+1;
        data_1(:,:,k) = data{i}.X((Pos(j)+2*250):(Pos(j)+6*250-1),1:22);
    end
end
clear data
% wipe off NaN
data_1(isnan(data_1)) = 0;

% E data
session_type = 'E';

load(strcat('D:\Projects\DataSet\BCICIV_2a_mat\A0',num2str(subject_index),session_type))
label_2 = [];
k = 0;
data_2 = zeros(1000,22,288);
for i = 4:9
    label_2 = [label_2;data{i}.y];
    Pos = data{i}.trial;
    for j = 1:numel(Pos)
        k = k+1;
        data_2(:,:,k) = data{i}.X((Pos(j)+2*250):(Pos(j)+6*250-1),1:22);
    end
end
clear data
% wipe off NaN
data_2(isnan(data_2)) = 0;

%% preprocessing
% option - band-pass filter
% fc = 250; % sampling rate
% Wl = 4; Wh = 36; % pass band
% Wn = [Wl*2 Wh*2]/fc;
% [b,a]=cheby2(6,60,Wn);
% 
% % a better filter for 4-40 Hz band-pass
% % fc = 250;
% % Wl = 4; Wh = 40; 
% % Wn = [Wl*2 Wh*2]/fc;
% % [b,a]=cheby2(8,20,Wn);
% 
% for j = 1:288
%     data_1(:,:,j) = filtfilt(b,a,data_1(:,:,j));
%     data_2(:,:,j) = filtfilt(b,a,data_2(:,:,j));
% end

% option - a simple standardization
%{
eeg_mean = mean(data,3);
eeg_std = std(data,1,3); 
fb_data = (data-eeg_mean)./eeg_std;
%}


%standarization
% RX1 = zeros(1000,1000,228);
% RX2 = zeros(1000,1000,228);
% for j = 1:288
%     X = data_1(:,:,j);
%     RX1(:,:,j) = X*X';
%     X = data_2(:,:,j);
%     RX2(:,:,j) = X*X';
% end
% MRX1 = mean(RX1,3);
% MRX2 = mean(RX2,3);
% for j = 1:288
%     data_1(:,:,j) = real(MRX1^(-1/2))*data_1(:,:,j);
%     data_2(:,:,j) = real(MRX2^(-1/2))*data_2(:,:,j);
% end

%% Save the data to a mat file 
data = data_1;
label = label_1;
% label = t_label + 1;
saveDir = ['D:\Projects\DataSet\standard_2a_data\rawA0',num2str(subject_index),'T.mat'];
save(saveDir,'data','label');

data = data_2;
label = label_2;
saveDir = ['D:\Projects\DataSet\standard_2a_data\rawA0',num2str(subject_index),'E.mat'];
save(saveDir,'data','label');
