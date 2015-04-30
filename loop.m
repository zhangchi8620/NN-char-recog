function loop (w,v,xx,r)
global hidden_Num input_Num output_Num eta w v;
o = zeros(1,output_Num);
delta_v = zeros (output_Num,hidden_Num + 1);
delta_w = zeros (hidden_Num + 1, input_Num + 1);
summ = 0;
x = [1, xx]';
z (1) = 1;
y = zeros(1,10);

for h = 2 : hidden_Num + 1
    for j = 1 : input_Num + 1
        summ = summ + w(h,j) * x(j);
    end
    z(h) = sigmoid(summ);
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

for i = 1 : output_Num
    for h = 1 : hidden_Num + 1
        delta_v(i,h) = eta * (r(i) - y(i)) * z(h);
    end
end


for h = 2 : hidden_Num + 1;
    for j = 1 : input_Num + 1;
        summ = 0;
        for i = 1 : output_Num
            summ = summ + (r(i) - y(i)) * v(i,h);
        end
        delta_w(h,j) = eta * summ * z(h) * (1-z(h)) * x(j);
    end
end

for i = 1 : output_Num
    for h = 1 : hidden_Num + 1
        v(i,h) = v(i,h) + delta_v(i,h);
    end
end

for h = 1 : hidden_Num + 1
    for j = 1 : input_Num + 1
        w(h,j) = w(h,j) + delta_w(h,j);
    end
end
%{
if (find(r==max(max(r))) == find(y==max(max(y))))
    correct = correct + 1;
else
    incorrect = incorrect + 1;
end

'(((('
 find(r==max(max(r)))
 find(y==max(max(y)))
 '))))'
%}