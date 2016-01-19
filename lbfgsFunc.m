function [opttheta, cost] = lbfgsFunc (visibleSize, hiddenSize, lambda, data, theta)
% Use L-BFGS algorithm to minimize a function
% This is function for you to use. You don't need to modify it.

%  Randomly initialize the parameters if not specified
if nargin < 5
    theta = initializeParameters(hiddenSize, visibleSize);
end

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'off';

[opttheta, cost] = minFunc( @(p) autoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, data), ...
                              theta, options);
end