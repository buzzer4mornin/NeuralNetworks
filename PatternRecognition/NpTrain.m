function [net,tr]=NpTrain(input,desire,hiddenLayerSize1,hiddenLayerSize2)
if(nargin==3)
    LayerSize=hiddenLayerSize1;
else
    LayerSize=[hiddenLayerSize1 hiddenLayerSize2];
end


% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by Neural Pattern Recognition app
% Created 12-Jan-2019 23:25:50
%
% This script assumes these variables are defined:
%
%   trainImgSmall - input data.
%   target - target data.

x = input;
t = desire;

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network

net = patternnet(LayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
%view(net)
% Train the Network
[net,tr] = train(net,x,t);


end


