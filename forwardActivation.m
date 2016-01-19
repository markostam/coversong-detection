function [a2, a3]=forwardActivation(W1, W2, b1, b2, X)
% Forward calculation of auto-encoder on a set (say k) of input examples
% The input and output layer each have n nodes and the hidden layer
% has m nodes.
%
% Input
%  - W1     : weights from input layer to hidden layer, (m*n) matrix
%  - W2     : weights from hidden layer to output layer, (n*m) matrix
%  - b1      : bias weights for the hidden layer, (m*1) vector
%  - b2      : bias weights for the output layer, (n*1) vector
%  - X       : input data, (n*k) matrix. Each column is one example. The number of
%               columns k is the number of examples. 
% Output
%  - a2      : output (activation) of the hidden layer, (m*k) matrix
%  - a3      : output (activation) of the output layer, (n*k) matrix
% 
% Author: Marko Stamenovic
% Created: 10/21/15
% Last modified: 10/23/2015
%---------------------------------------------------------

%sigmoid f(x)=1/(1+exp(-x))

a2 = W1*X+b1*ones(1,size(X,2)); %applies weight W1 and adds a copy of bias b1 to each column of a2
a2 = (1+exp((-1*a2))).^-1; % normalize with sig function and weighting

a3 = W2*a2+b2*ones(1,size(X,2)); %applies weight W2 and adds a copy of bias b1 to each column of a3
a3 = (1+exp((-1*a3))).^-1; % normalize with sig function and weighting
        
%---------------------------------------------------------
end
