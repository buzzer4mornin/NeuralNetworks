function acc=accuracy(y,desire)
corr=(y==desire);
acc=sum(corr)/length(y);
end