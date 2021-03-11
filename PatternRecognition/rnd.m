function x=rnd(y)
%a value less then 0 interpretad as 0
y(y<0)=0;
%a value greater then 9 interpretad as 9
y(y>9)=9;
% a value between zero and 9 rounded to nearest integer
x=round(y);
end