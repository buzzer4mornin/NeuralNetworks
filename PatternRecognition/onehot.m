function x=onehot(input)
I=eye(max(input)+1);
%I(:,1)=0;
enc=zeros(max(input)+1,length(input));

    for i=1:length(input)
%         if(input(i)==0)
%             enc(:,i)=zeros(max(input),1);
%         else
            enc(:,i)=I(:,input(i)+1);
%         end
    end
    x=enc;
end