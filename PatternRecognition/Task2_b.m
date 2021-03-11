%Load data
clear
load FMNISTSmall
%%
%converting target 
target=onehot(trainLabelSmall);
%%
hiddenLayer1=0;
hiddenLayer2=0;
old_accuracy=0;

for i=50:5:200
    hiddenLayerSize1=i;
    for j=105:5:200
        hiddenLayerSize2=j;
        [netNp,tr]=NpTrain(trainImgSmall,target,hiddenLayerSize1,hiddenLayerSize2);

        % Test the Network
        y = netNp(testImgSmall);
        %convert vec to indices
        predicted=vec2ind(y);
        % subtracting 1 to get real labels [0 9]
        predicted=predicted-1;
        nntraintool close
        %calculating accuracy by computing correctly classified
        %labels
        acc=accuracy(predicted,testLabelSmall);
        fprintf('Neurons in first : %d & in second hidden layer: %d. Accuracy : %d \n', i,j, acc*100);
        if(acc>old_accuracy)
            new_nprt=netNp;
            old_accuracy=acc;
            hiddenLayer1=i;
            hiddenLayer2=j;
        end
    end
end

%%
Labels={'0 - T-shirt/top','1 - Trousers','2 - Pullover','3 - Dress','4 - Coat','5 - Sandal','6 - Shirt','7 - Sneaker','8 - Bag','9 - Ankle boot'};

y = Nprt_2Layer(testImgSmall);

%convert vec to indices
pred=vec2ind(y);
% subtracting 1 to get real labels [0 9]
pred=pred-1;
%indices where our net is incorrectly classifying the labels
indices=find(pred~=testLabelSmall);
%selecting randomly sample of 10 incorrect classified labels without
%replacement
sample = datasample(indices,10,'Replace',false);

for i=1:length(sample)
    label=testLabelSmall(:,sample(:,i));
    tlabel=Labels{label+1}; % true label
    figure 
    ax(1) = subplot(1,2,1); 
    colormap(gray)
    I=reshape(testImgSmall(:,sample(:,i)),28,28);
    image(I);title(['True Label : ',tlabel])
    ax(2) = subplot(122); 
    label=pred(:,sample(:,i));
    plabel=Labels{label+1};
    %Predicted Label;
    I=reshape(testImgSmall(:,sample(:,i)),28,28);
    image(I); title(['Predicted Label : ',plabel])
end
%%
%plot confusion matrix
t=onehot(testLabelSmall);
x=onehot(pred);
plotconfusion(t,x)

