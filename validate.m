function [err,correct, result] = validate (w,v,xx,r)
global hidden_Num input_Num output_Num eta;
x = [1, xx];
z (1) = 1;
y = zeros(1,10);
summ = 0;
o = zeros(1,output_Num);

for h = 2 : hidden_Num + 1
    for j = 1 : input_Num + 1
        summ = summ + w(h,j) * x(j);
    end
    z(h) = sigmoid (summ);
end


total = 0;
for i = 1 : output_Num
    for h = 1 : hidden_Num + 1
        o(i) = o(i) + v(i,h) * z(h);
    end
    total = total + exp(o(i));
end

for i = 1: output_Num
    y(i) = exp(o(i))/total;
end

err = (r - y).^2 ;

if (find(r==max(max(r))) == find(y==max(max(y))))
    correct = 1;
else
    correct = 0;
end

result = find(y==max(max(y)));
