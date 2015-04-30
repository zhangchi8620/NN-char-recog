
load('hidden_4.mat');
plot(Error_valid(1:399),'-r','LineWidth',2);hold on;
load('hidden_10.mat');
plot(Error_valid,'-b','LineWidth',2);hold on;
load('hidden_15.mat');
plot(Error_valid,'-g','LineWidth',2);hold on;
load('hidden_20.mat');
plot(Error_valid,'-k','LineWidth',2);hold on;

h = legend('Hidden units No. = 4','Hidden units No. = 10','Hidden units No. = 15','Hidden units No. = 20',epoch);
     set(h,'Interpreter','none')