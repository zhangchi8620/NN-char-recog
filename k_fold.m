%%
clear;
warning off all;
%path = 'J:\zc\MATLAB\data\learn-grid\';
str = 'acdefghlpr';
path = '/home/parker/courses/cs425-528/Project1/';

learn_path = strcat(path,'learn-grid/');
valid_path = strcat(path,'validate-grid/');
test_path = strcat(path,'test-grid/');

global hidden_Num input_Num output_Num eta learn_Num valid_Num w v;
%initialize 

learn_Num = 100;    % number of learn set poionts
valid_Num = 250;    % number of valid set points
test_Num = 250;     % number of test set points

input_Num = 96;     % number of input nodes: 12*8 = 96
hidden_Num = 10;     % number of hidden nodes
output_Num = 10;    % number of output nodes

index_kfold = 1;
index_test = 1;

% loading data
for i=1:10
   for j=1:1:learn_Num
       filename = [learn_path,str(i),int2str(j)];
       matrix = textread(filename);
	   kfold(i,j).content = matrix(1:12, 1:8);
	   kfold(i,j).tag = matrix(13,1:10);
   end
   for j=1:1:valid_Num
       filename = [valid_path,str(i),int2str(j)];
       matrix = textread(filename);
	   kfold(i,j+100).content = matrix(1:12, 1:8);
	   kfold(i,j+100).tag = matrix(13,1:10);
   end
end
for i=1:10
   for j=251:1:test_Num+250;
       filename = [test_path,str(i),int2str(j)];
       matrix = textread(filename);
	   test(index_test).content = matrix(1:12, 1:8);
	   test(index_test).tag = matrix(13,1:10);
       index_test=index_test+1;
   end
end

seq = randperm(350);

for i = 1 : 350
    for j= 1: 10
        temp(j,i) = kfold(j,seq(i));
    end
end

Error_valid = 0;
Error_learn = 0;
Error_test = 0;
Cor_learn = zeros(1,20000);
Cor_valid = zeros(1,20000);

Node_err = zeros(20000,10);

rate_learn = zeros(1,20000);
rate_valid = zeros(1,20000);
error_sum = zeros(1,20000);

eta = 0.1;
epoch = 0;

w = random('unif',-0.01,0.01,hidden_Num+1, input_Num + 1);
v = random('unif',-0.01,0.01,output_Num, hidden_Num + 1);

format long;
RUN = 1;

tic;
 fprintf('<<<<<<<<<< Start. k-fold cross validation.\n');
while (RUN) 
    epoch = epoch + 1;
    W = w;
    V = v;
    for t=1:10  % Re-divide learn and valid sets
    nn = circshift(temp,[0,-1]);
        for i= 1:35
            for j = 1:10
                valid_t(j,i) = nn(j,i);
            end
        end
        for i=36:350
            for j=1:10
                learn_t(j,i-35) = nn(j,i);
            end
        end
        valid = reshape(valid_t,1,350);
        learn = reshape(learn_t,1,3150);

        % Train
        order = randperm(3150);  
        W = w;
        V = v;
        
        node_sum = 0;
        cor_learn= 0;
        for i = 1:3150     % train each input node
            index = order(i);
            loop(w,v,reshape(learn(index).content',1,96), reshape(learn(index).tag',1,10));
            [err_learn, cor_learn] = validate(w,v,reshape(learn(index).content',1,96), reshape(learn(index).tag',1,10));
            Cor_learn(epoch) = Cor_learn(epoch) + cor_learn;
            node_sum = node_sum + err_learn;
        end
        Node_err(epoch,:) = 0.5 * node_sum' /(315 * 10);
        Error_learn(epoch) = 0.5 * sum(node_sum' /(315 * 10));
        rate_learn(epoch) = Cor_learn(epoch) / (315 * 10);

        node_sum = 0;
        cor_valid = 0;
        for j = 1:350
            [err_valid, cor_valid] = validate(w, v, reshape(valid(j).content',1,96), reshape(valid(j).tag',1,10)); 
            Cor_valid(epoch) = Cor_valid(epoch) + cor_valid;
            node_sum = node_sum + err_valid;
        end

        Error_valid_node(epoch, :) = 0.5 * node_sum' / (35 * 10);
        Error_valid(epoch) = 0.5 * sum (node_sum' / (35 * 10));
        rate_valid(epoch) = Cor_valid(epoch) / (35 * 10);
        
        error_sum(epoch) = error_sum(epoch) + Error_valid(epoch); 
    end
	error_sum(epoch) = error_sum(epoch) /10;
	if (epoch > 1 && error_sum(epoch-1)<0.04)
        if (error_sum(epoch) >= error_sum(epoch - 1))
            RUN = 0;      
        end
    end
end


% Train for last time with all data
all = reshape(nn,1,3500);
for i = 1:3500
        loop(W,V,reshape(all(i).content',1,96), reshape(all(i).tag',1,10));
end

time_learn = toc;
Cor_test_sum = 0;
Cor_test = zeros(1, 10);
node_sum = 0;
rate_test = 0;
for j =1 : test_Num *10
    mm = reshape(test(j).tag',1,10);
    [err_test, cor_test] = validate(w,v,reshape(test(j).content',1,96), mm);
    node_sum = node_sum + err_test;
    Cor_test_sum = Cor_test_sum + cor_test;
	Cor_test(find(test(j).tag)) = Cor_test(find(test(j).tag)) + cor_test;

    Error_test(j) = 0.5 * sum (node_sum' / (test_Num * 10));
end
rate_test_sum = Cor_test_sum / (test_Num * 10);
rate_test = Cor_test / test_Num ;

for j = 1 : 10
	fprintf('Success rate for character %s is %f\n',str(j), rate_test(j));
end

fprintf('Total rate is %d\n', rate_test_sum);
fprintf('Training time is %f\n', time_learn);
