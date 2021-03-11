clear
clc
%Load data
load FMNISTSmall

testAccuracy=0;
%%
for i=20:20
    hiddenLayerSize=i;
%Train feed forward network
[net,tr]=Train(trainImgSmall,trainLabelSmall,hiddenLayerSize);

% Test the Network
y=net(testImgSmall);

%transforming output values [0 9]
y=rnd(y);
%converting to one hot encoding for plotting confusion matrix
predicted=onehot(y);
desire=onehot(testLabelSmall);

% calculating accuracy
acc=accuracy(y,testLabelSmall);
fprintf('Neurons in hidden layer: %d and aaccuracy : %d\n', i, acc*100);
    if(acc>=testAccuracy)
        testAccuracy=acc;
        hiddenLayerSize=i;
        Nft_net=net;
    end
end
nntraintool close
%%
y=net(testImgSmall);

%transforming output values [0 9]
y=rnd(y);
%converting to one hot encoding for plotting confusion matrix
 predicted=onehot(y);
 desire=onehot(testLabelSmall);
% 
% % calculating accuracy
% acc=accuracy(y,testLabelSmall);
plotconfusion(desire,predicted)
%y1=net(testLabelSmall);

