function [cost,grad] = autoencoderCost(theta, visibleSize, hiddenSize, lambda, X)
% This function calcualtes the overall cost of an auto-encoder on all input
% data, and the partial derivatives of the cost w.r.t all weights.
%
% Input
%  - theta          : all weights arranged as a vector. Its length should be
%                       2*visibleSize*hiddenSize (for W1 and W2) +
%                       hiddenSize (for b1) + visibleSize (for b2)
%  - visibleSize    : input layer size
%  - hiddenSize   : hidden layer size
%  - lambda        : parameter for the regularization term
%  - X               : the input data matrix. Each column is an example
% 
% Output
%  - cost           : the overal error cost J(W,b) that we want to
%                       minimize. A scalar
%  - grad           : the gradients of the cost w.r.t. to all weights. A
%                       vector of the same size as theta.
%
% Author: Marko Stamenovic
% Created: 10/22/15
% Last modified: 10/28/2015

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables, we initialize them to zeros.
cost = 0;
grad = zeros(size(theta));
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
%----------------------------------------------

%compute activations
[a2, a3]=forwardActivation(W1, W2, b1, b2, X);
m = size(X,2); %good

%compute error term

%output layer S3
delta_nl = (a3-X).*(a3-a3.^2); 
%hidden layer S2
delta_1 = (W2'*delta_nl).*(a2-a2.^2); 

%compute gradients
W1grad = delta_1*(X')./m+lambda.*W1;
W2grad = delta_nl*(a2')./m+lambda.*W2;
b1grad = mean(delta_1,2);
b2grad = mean(delta_nl,2);

%h(wb)(x) = a3

%calculate cost via squared error & reg term 
squared_error = mean(sum((a3-X).^2))/2; 
reg = (lambda/2)*(sum(sum(W1.^2))+sum(sum(W2.^2))); 
cost = squared_error+reg;

%-------------------------------------------
% After computing the cost and gradient, we convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% the gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end