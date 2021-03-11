%Load data
clear
load FMNISTSmall
%%
%converting target 
target=onehot(trainLabelSmall);
%%
hiddenLayer1=0;
old_accuracy=86;
    for j=60:5:200
        hiddenLayerSize1=j;
        [netNp,tr]=NpTrain(trainImgSmall,target,hiddenLayerSize1);
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
        fprintf('Neurons in hidden layer: %d. Accuracy : %d \n',j, acc*100);
        if(acc>=old_accuracy)
            Nprt_1Layer=netNp;
            old_accuracy=acc;
            hiddenLayer1=j;
        end
    end
    %%
        y = Nprt_1Layer(testImgSmall);
        %convert vec to indices
        predicted=vec2ind(y);
        % subtracting 1 to get real labels [0 9]
        predicted=predicted-1;
        t=onehot(testLabelSmall);
        x=onehot(predicted);
    %plotting confusion matrix
    plotconfusion(t,x);
    