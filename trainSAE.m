function [W1, b1, W2, b2] = trainSAE(visibleSize,hiddenSizeL1,hiddenSizeL2,lambda,X)
% Function to train a stacked auto-encoder (SAE) with 2 hidden layers
%
% Input
%  - visibleSize     : size of input layer
%  - hiddenSizeL1 : size of 1st hidden layer 
%  - hiddenSizeL2 : size of 2nd hidden layer
%  - lambda         : regularization parameter
%  - X                : input data matrix
% 
% Output
%  - W1             : trained weights to the first hidden layer
%  - b1              : trained biases to the first hidden layer
%  - W2             : trained weights to the second hidden layer
%  - b2              : trained biases to the second hidden layer
% Marko Stamenovic
% 10/28/2015

%  initialize parameters parameters 
theta1 = initializeParameters(hiddenSizeL1, visibleSize);

%-----------------------------------------

% train the first auto-encoder

[opttheta1, cost] = lbfgsFunc(visibleSize, hiddenSizeL1, lambda, X, theta1);

W1L1 = reshape(opttheta1(1:hiddenSizeL1*visibleSize), hiddenSizeL1, visibleSize);
W2L1 = reshape(opttheta1(hiddenSizeL1*visibleSize+1:2*hiddenSizeL1*visibleSize), visibleSize, hiddenSizeL1);
b1L1 = opttheta1(2*hiddenSizeL1*visibleSize+1:2*hiddenSizeL1*visibleSize+hiddenSizeL1);
b2L1 = opttheta1(2*hiddenSizeL1*visibleSize+hiddenSizeL1+1:end);
save('W1b1L1.mat','W1L1','b1L1')

%-----------------------------------------

%  initialize parameters

theta2 = initializeParameters(hiddenSizeL2, hiddenSizeL1);
[a2, a3]=forwardActivation(W1L1, W2L1, b1L1, b2L1, X)

%-----------------------------------------

% train the second auto-encoder

[opttheta2, cost] = lbfgsFunc (hiddenSizeL1, hiddenSizeL2, lambda, a2, theta2);

W1L2 = reshape(opttheta2(1:hiddenSizeL2*hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
W2L2 = reshape(opttheta2(hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1), hiddenSizeL1, hiddenSizeL2);
b1L2 = opttheta2(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);
b2L2 = opttheta2(2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2+1:end);
save('W1b1L2.mat','W1L2','b1L2')

%-----------------------------------------

W1 = W1L1;
b1 = b1L1;
W2 = W1L2;
b2 = b1L2;

%-- end --
