clear 

addpath tool/
addpath tool/tf_agc
addpath minFunc

% intiation
fs = 16000;
Imitation_file=dir('covers_pairs/*.wav');
L_Imitation=length(Imitation_file);
cellrecording=cell(L_Imitation,1);
numSum=0;

for i=1:1:L_Imitation

    wav_Imitation=audioread(fullfile('covers_pairs',Imitation_file(i).name));
    Imitation_file(i).name
    b=beat2(wav_Imitation,fs)*fs;                                                             %extract beats from sample
    for lenb=1:length(b)-1
        b_min(lenb) = (b(lenb+1)-b(lenb));                                                  %get min of beat sample difference
    end
    b_min = floor(min(b_min));
    b_min_list(i)=b_min;
    wav_Imitation = wav_Imitation(b(1):b(end));                                      %truncates wav file to first beat and last beat
    
    if mod(b_min,2) ~= 0                                                                            %makes sure beat interval is even for smooth CQT
        b_min=b_min+1;
    end
    L_win=b_min;                                                                                        %window size (patch size) = 1/4th note
    L_hop=floor(L_win/2);                                                                            % overlap: hop size is 1/8th note
    len_wav=length(wav_Imitation);                                                          %length of wav file
    intCQT = logfsgram(wav_Imitation, L_win, fs, L_win, L_hop,21,21);   %180 bin cqt ==> for 8 frame patch
    
    %patches
    patch_win=8;                                                                                           %number of cqt windows per patch
    len=size(intCQT,2);                                                                                  %number of elements along x axis of spectrogram
    num=floor(len/(patch_win/2));                                                              %number of patches. determines hop size. we have #patches = beats. so patch hop size = 1/4 note. 
    numSum=numSum+num;
    cellpatch=cell(num,1);
    index=1:(len-patch_win);
    for j=1:1:num
        cellpatch(j)={intCQT(:,index(j):index(j)+patch_win-1)};
    end
    patch1=cell2mat(cellpatch);
    patch1=round(255*patch1/max(max(patch1)));
    cellrecording(i)={patch1};
end

recording=cell2mat(cellrecording);

% Remove silence patches
[M,N]=size(recording);
num=M/180;
count=0;
for i=1:num
    patch=recording((i-1)*180+1:i*180,:);
    if rms(rms(patch))>0.01
        count=count+1;
    end
end
numAftRemv=count;
cellpatch=cell(numAftRemv,1);

count=0;
for i =1:num
    patch=recording((i-1)*180+1:i*180,:);
    if rms(rms(patch))>0.01
        count=count+1;
        cellpatch(count)={patch};
    end
end

recordingAftRemv=cell2mat(cellpatch);
save('dataAftRemv.mat','recordingAftRemv', '-v7.3');

% train the SAE
visibleSize=1440; % number of input neurons of the 1st AE 
hiddenSizeL1=500;   % number of hidden neurons of the 1st AE 
hiddenSizeL2=100;   % number of hidden neurons of the 2nd AE
% sparsityParam=0.01; % desired average activation of the hidden units.
%                      % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p"). 
lambda=0.0001;  % weight decay parameter  
% beta=3; % weight of sparsity penalty term
patches = sampleIMAGES_AftRemv2;
trainSAE(visibleSize,hiddenSizeL1,hiddenSizeL2,lambda,patches);

test_final_allsongs_cqt %test SAE