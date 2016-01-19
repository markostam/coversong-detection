function patches = sampleIMAGES_AftRemv()
% sampleIMAGES
% Returns a number of patches for training

load dataAftRemv;    % load images from disk 

patchsize =180*8;  % we'll use 8x8 patches 
[m,n]=size(recordingAftRemv);
m=round(m);
n=round(n);
numpatches=floor(m/180);

% Initialize patches with zeros.  
patches = zeros(patchsize, numpatches);


for i=1:numpatches
    patches(:,i)=reshape(recordingAftRemv((i-1)*180+1:i*180,:),[patchsize,1]);
end

% For the autoencoder to work data must be normalized
% Since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function),  
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Normalize data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end

