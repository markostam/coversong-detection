clear;

addpath tool/
addpath tool/tf_agc
addpath minFunc

load W1b1L1;
load W1b1L2;

% init values for CQT
fs = 16000;
bins_per_octave = 21;

% extract features from the ground truth recordings
gtFiles=dir('covers_pairs/*.wav');
L_gt=length(gtFiles);                                                           % number of files in folder
patchInAllFiles=cell(L_gt,1);                                               % each cell represents all the patches in one .wav file
numOfPatchInOneFile=zeros(L_gt,1);                                % each element represents the number of patches in one .wav file
numSum=0;

for i=1:L_gt
    wav_gt=audioread(fullfile('covers_pairs',gtFiles(i).name));
    gtFiles(i).name
    b=beat2(wav_gt,fs)*fs;                                                  %extract beats from sample
    for lenb=1:length(b)-1
        b_min(lenb) = (b(lenb+1)-b(lenb));                          %get min of beat sample difference
    end
    b_min = floor(min(b_min));
    wav_gt = wav_gt(b(1):end);                                          %truncates wav file to first beat
    
    if mod(b_min,2) ~= 0                                                    %ensure beat interval is even for smooth CQT
        b_min=b_min+1;
    end
    L_win=b_min;                                                                 %window size (patch size) = 1/4th note
    L_hop=L_win/2;                                                              % hop size is 1/8th note
    len_wav=length(wav_gt);                                              %length of wav file
    intCQT_gt = logfsgram(wav_gt, L_win, fs, L_win, L_hop,21,21);    %180 bin cqt ==> for 8 frame patch

    patch_win=8;                                                                %number of cqt windows per patch
    len=size(intCQT_gt,2);                                                 %number of elements along x axis of spectrogram
    num=floor(len/(patch_win/w));                                  %number of patches. determines hop size. #patches = #beats, so hop size = 1/4 note.
    numSum=numSum+num;
    cellpatch=cell(num,1);
    numOfPatchInOneFile(i)=num;                                    % number of patches in one .wav file stored
    
    for k=1:num-patch_win                                               % reshape the patches into long vectors of 1*1440
        intCQT_patch=intCQT_gt(:,(k-1)+1:(k-1)+patch_win);
        intCQT_patch=reshape(intCQT_patch,[180*patch_win,1]);
        intCQT_patch=intCQT_patch';
        cellpatch(k)={intCQT_patch};
    end
    
    patchInOneFile=cell2mat(cellpatch);
    patchInOneFile=round(255*patchInOneFile/max(max(patchInOneFile))); % normalization of the patches in one .wav file
    
    % silence removal by rms detection
    cout=0;
    for l=1:size(patchInOneFile,1)
        if rms(rms(patchInOneFile(l,:)))>=0.01
            cout=cout+1;
        end
    end
    numPatchAftRmv=cout;
    patchInOneFile_silence_rmv=cell(numPatchAftRmv,1);
    
    cout=0;
    for l=1:size(patchInOneFile,1)
        if rms(rms(patchInOneFile(l,:)))>=0.01
            cout=cout+1;
            patchInOneFile_silence_rmv(cout)={patchInOneFile(l,:)};
        end
    end
    patchInAllFiles(i)={cell2mat(patchInOneFile_silence_rmv)};

end
    
    
%%%%%%%%%%%%%%%RESULTS CALCULATIONS%%%%%%%%%%%%%%%%%

% initialize results tables
results_dtw_allsongs = zeros(L_gt, L_gt);
results_dtwnorm_allsongs = zeros(L_gt, L_gt);
results_euclidean_allsongs = zeros(L_gt, L_gt);
results_xcorr_allsongs = zeros(L_gt, L_gt);
results_kldiv_allsongs = zeros(L_gt, L_gt);
results_mahaldist_allsongs = zeros(L_gt, L_gt);
counter=0

%calculate output features
for i = 1:L_gt
    a2 = W1L1*patchInAllFiles{i,1}'+b1L1*ones(1,size(patchInAllFiles{i,1}',2)); %applies weight W1 and adds a copy of bias b1 to each column of a2
    a2 = (1+exp((-1*a2))).^-1; % normalize 1st hiddn layer with sig function and weighting
    
    a3 = W1L2*a2+b1L2*ones(1,size(patchInAllFiles{i,1}',2)); %applies weight W2 and adds a copy of bias b1 to each column of a3
    a3 = (1+exp((-1*a3))).^-1; % normalize 2nd hidden layer (output layer) with sig function and weighting
    
    a3_cells{i}=a3;
    a3_cells_mean{i}=mean(a3,2);
    fprintf('>');
    if mod(i,20)==0
        fprintf('\n');
    end
    
end
fprintf('output features done');


%calculate results
for n = 1:L_gt
    for i = 1:L_gt

        % Distance between sampled song and sampling song .
        %results_dtw_allsongs(n,i)=dtw(a3_cells{n}',a3_cells{i}');
        results_dtw_allsongs(n,i)=dtw(a3_cells{n}',a3_cells{i}');                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           results_dtwnorm_allsongs(n,i)=dtw(a3_cells{n}',a3_cells{i}')/(size(a3_cells{n},2)+size(a3_cells{i},2)); %normalized over length of output patches
        %DTW TAKES A LONG TIME BUT IS MOST ACCURATE. CONSIDER COMMENTING IT
        %OUT FOR QUICK AND DIRTY RESULTS
        
        %educlidean distance
        results_euclidean_allsongs(n,i)=sum(abs(a3_cells_mean{n}-a3_cells_mean{i}).^2);
        
        %cross correlation
        results_xcorr_allsongs(n,i)=sum(xcorr(a3_cells_mean{n}-a3_cells_mean{i}));
        
        %kl divergence
        results_kldiv_allsongs(n,i)=sum(KLDiv(a3_cells_mean{n}',a3_cells_mean{i}'));        
        
        %mahalanobis distance
        %results_mahaldist_allsongs(n,i)=mean(mean(pdist2(a3_cells{n}',a3_cells{i}','mahalanobis'),2),1);
        fprintf('>');
        if mod(i,20)==0
            fprintf('\n');
        end
    end
   fprintf('\n');
end

%set results diagonals to high values so they are ignored by min command
for i = 1:L_gt
    results_dtwnorm_allsongs(i,i)=9999999;
    results_euclidean_allsongs(i,i)=9999999;
    results_kldiv_allsongs(i,i)=9999999;
    results_xcorr_allsongs(i,i)=9999999;
end

%calculate overall results
correct_guesses=zeros(3,4);

for i = 1:2:L_gt %covers vs originals
    if results_dtwnorm_allsongs(i,i+1) <= min(results_dtwnorm_allsongs(i,:));
        correct_guesses(2,1)=correct_guesses(2,1)+1;
    end
    
    if results_euclidean_allsongs(i,i+1) <= min(results_euclidean_allsongs(i,:));
        correct_guesses(2,2)=correct_guesses(2,2)+1;
    end
    
    if results_kldiv_allsongs(i,i+1) <= min(results_kldiv_allsongs(i,:));
        correct_guesses(2,3)=correct_guesses(2,3)+1;
    end
    
    if results_xcorr_allsongs(i,i+1) <= min(results_xcorr_allsongs(i,:));
        correct_guesses(2,4)=correct_guesses(2,4)+1;
    end
    
    
end

for i = 2:2:L_gt %originals vs covers
    if results_dtwnorm_allsongs(i,i-1) <= min(results_dtwnorm_allsongs(i,:));
        correct_guesses(3,1)=correct_guesses(3,1)+1;
    end
    
    if results_euclidean_allsongs(i,i-1) <= min(results_euclidean_allsongs(i,:));
        correct_guesses(3,2)=correct_guesses(3,2)+1;
    end
    
    if results_kldiv_allsongs(i,i-1) <= min(results_kldiv_allsongs(i,:));
        correct_guesses(3,3)=correct_guesses(3,3)+1;
    end
    
    if results_xcorr_allsongs(i,i-1) <= min(results_xcorr_allsongs(i,:));
        correct_guesses(3,4)=correct_guesses(3,4)+1;
    end
end

%correct_guesses(1,1)='results_dtwnorm_allsongs';
%correct_guesses(1,2)='results_euclidean_allsongs';
%correct_guesses(1,3)='results_kldiv_allsongs';
%correct_guesses(1,4)='results_xcorr_allsongs';


correct_guesses %distance from training set to covers
%should be the same for all but KL divergence