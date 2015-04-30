%% 
clear;
warning off all;
global hidden_Num input_Num output_Num eta learn_Num valid_Num w v;
path = '/home/parker/courses/cs425-528/Project1/';

while true
    mode = input('Please select mode: training, testing, exit\n', 's');
    if(strcmp(mode, 'training'))
        fprintf('<<<<<<<<<< Now in training mode\n');    

        eta = input('Please input learning rate:\n');
        hidden_Num = input('Please input hidden units number: \n');

        index_learn = 1;    
        index_valid = 1;
        index_test = 1;

        learn_Num = 100;    % number of learn set poionts
        valid_Num = 250;    % number of valid set points
        test_Num = 250;     % number of test set points

        input_Num = 96;     % number of input nodes: 12*8 = 96
        output_Num = 10;    % number of output nodes

        TIMES = 1;
        rate_test = zeros(TIMES,10);
        time_learn = zeros(TIMES,1);

        % Loading data
        fprintf('Loading data...\n');
        str = 'acdefghlpr';
        learn_path = strcat(path, 'learn-grid/');
        valid_path = strcat(path, 'validate-grid/');
        test_path = strcat(path, 'test-grid/');
        for i=1:10
           for j=1:1:learn_Num
               filename = [learn_path,str(i),int2str(j)];
               matrix = textread(filename);
               learn(index_learn).content = matrix(1:12, 1:8);
               learn(index_learn).tag = matrix(13,1:10);
               index_learn=index_learn+1;
           end
        end

        for i=1:10
           for j=1:1:valid_Num;
               filename = [valid_path,str(i),int2str(j)];
               matrix = textread(filename);
               valid(index_valid).content = matrix(1:12, 1:8);
               valid(index_valid).tag = matrix(13,1:10);
               index_valid=index_valid+1;
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

        for Times = 1:TIMES
            %fprintf('Training for %d times\n',Times);
            fprintf('Training ANN %d ...\n', Times);
            %initialize 
            w = random('unif',-0.01,0.01,hidden_Num+1, input_Num + 1);
            v = random('unif',-0.01,0.01,output_Num, hidden_Num + 1);

            Error_valid = 0;
            Error_learn = 0;
            Error_test = 0;
            Cor_learn = zeros(1,20000);
            Cor_valid = zeros(1,20000);

            Node_err = zeros(20000,10);

            rate_learn = zeros(1,20000);
            rate_valid = zeros(1,20000);

            epoch = 0;

            RUN = 1;
            format long;

            % Training and validating
            tic;
            while (RUN) 
                order = randperm(learn_Num * 10);  
                W = w;
                V = v;
                epoch = epoch + 1;
                node_sum = 0;
                cor_learn= 0;
                for i = 1:learn_Num * 10     % train each input node
                    index = order(i);
                    loop(w,v,reshape(learn(index).content',1,96), reshape(learn(index).tag',1,10));
                    [err_learn, cor_learn] = validate(w,v,reshape(learn(index).content',1,96), reshape(learn(index).tag',1,10));
                    Cor_learn(epoch) = Cor_learn(epoch) + cor_learn;
                    node_sum = node_sum + err_learn;
                end

                Node_err(epoch,:) = 0.5 * node_sum' / (learn_Num * 10);
                Error_learn(epoch) = 0.5 * sum(node_sum') / (learn_Num * 10);
                rate_learn(epoch) = Cor_learn(epoch) / (learn_Num * 10);

                node_sum = 0;
                cor_valid = 0;
                for j = 1:valid_Num * 10
                    [err_valid, cor_valid] = validate(w, v, reshape(valid(j).content',1,96), reshape(valid(j).tag',1,10));
                    node_sum = node_sum + err_valid;
                    Cor_valid(epoch) = Cor_valid(epoch) + cor_valid;
                end

                Error_valid_node(epoch,:) = 0.5 * node_sum' / (valid_Num * 10);
                Error_valid(epoch) = 0.5 * sum (node_sum') / (valid_Num * 10);
                rate_valid(epoch) = Cor_valid(epoch) / (valid_Num * 10);

				if (epoch >1 && Error_valid(epoch-1)<0.08)
                    if (Error_valid(epoch) >= Error_valid(epoch - 1))
                        RUN = 0;
						fprintf('Total training iterations: %d\n', epoch-1);
                    end
                end
            end

            time_learn(Times) = toc;


            %Ploting error-epoch
            %{
                figure();
                plot(Error_valid(1:epoch),'+r','MarkerSize',10);hold on;
                plot(Error_learn(1:epoch),'*b','MarkerSize',10); 
                axis([1 epoch 0 0.3])
                xlabel('Epoch') 
                %title('Error versus weight updates(epoch)')
                ylabel('Error')
                 h = legend('Validation set error','Training set error',epoch);
                 set(h,'Interpreter','none')

                figure();
                for node=1:10
                    plot(Node_err(1:epoch,node),'LineWidth',2); hold on;
                end
                axis([1 epoch 0 0.05]);
                xlabel('Epoch')
                %title('Sum of squared errors for each output unit');
                ylabel('Error')
             %}

            %Testing 
            node_sum = 0;
            Cor_test = zeros(1,10);
			confusion_test = zeros(10, 10);
            for j =1 : test_Num *10
                mm = reshape(test(j).tag',1,10);
                [err_test, cor_test, result] = validate(W,V,reshape(test(j).content',1,96), mm);
                node_sum = node_sum + err_test;
                Cor_test(find(test(j).tag)) = Cor_test(find(test(j).tag)) + cor_test;
                Error_test(j) = 0.5 * sum (node_sum' / (test_Num * 10));
				confusion_test(find(test(j).tag), result) =confusion_test(find(test(j).tag), result) + 1;
            end
            rate_test(Times,:) = Cor_test / test_Num ;
            for j = 1 : 10
                fprintf('Success rate for character %s is %f\n',str(j), rate_test(Times,j));
            end
			fprintf('Confusion Matrix:');
			display(confusion_test);
			
			fprintf('Average classification rate (testing set): %d\n', sum(rate_test(Times, :))/10);
			fprintf('Training time is %d\n', time_learn(Times));
		end

        rate_test_overall = sum(sum(rate_test)) /10 / TIMES;
		time_learn_overall = sum(time_learn) / TIMES;
        %fprintf('----Overall----\nAverage classification rate (testing set): %d\n', rate_test_overall);
        %fprintf('Training time is %d\n', time_learn_overall);
		%Saving data
        %{
        save('Err_eta_0.8.mat','Error_valid');
        save('w_10_times.mat','W');
        save('v_10_times.mat','V');
        save('Error_valid_10_times0.mat','Error_valid');
        save('Error_learn_10_times.mat','Error_learn');
        save('rate_10_times.mat','rate_test');
        save('time_10_times.mat','time_learn');
        save('RATE_char.mat','RATE');
        %}
        
    elseif(strcmp(mode, 'testing'))
        fprintf('<<<<<<<<<< Now in testing mode\n');
        
        while true
            testfile = input('Please type "folder/file_name" or "exit":', 's');
			filename = strcat(path, testfile);
			if (strcmp(testfile, 'exit'))
                break;
			end
			try
				matrix_test = textread(filename);
			catch exception
				fprintf('Invalid file. Please follow the format folder/file_name and try again.\nNote that files in test-grid folder starts from 251.\n');
				break;
			end
			
			ans = importdata('best.mat');
			W = ans.W;
            V = ans.V;
            hidden_Num = ans.hidden_Num;
            input_Num = ans.input_Num;
            output_Num = ans.output_Num;
            eta = ans.eta;
 
            test_test.content = matrix_test(1:12, 1:8);
            test_test.tag = matrix_test(13,1:10);                
            mm = reshape(test_test.tag',1,10);
            [err_test, cor_test, test_result] = validate(W,V,reshape(test_test.content',1,96), mm);
            fprintf('ANN test the character is ');
            switch test_result
                case 1
                    disp('a');
                case 2
                    disp('c');
                case 3
                    disp('d');
                case 4
                    disp('e');
                case 5
                    disp('f');
                case 6
                    disp('g');
                case 7
                    disp('h');
                case 8
                    disp('l');
                case 9
                    disp('p');
                case 10
                    disp('r');
                otherwise
                    disp('no such character.');
            end   
		end            
    elseif(strcmp(mode, 'exit'))
		break;
	else
        fprintf('Error input. Please try again.\n');
    end
end
