
load('Err_eta_0.01.mat');
plot(Error_valid,'-r','LineWidth',2);hold on;
load('Err_eta_0.05.mat');
plot(Error_valid,'-b','LineWidth',2);hold on;
load('Err_eta_0.1.mat');
plot(Error_valid,'-g','LineWidth',2);hold on;
load('Err_eta_0.5.mat');
plot(Error_valid,'-k','LineWidth',2);hold on;
load('Err_eta_0.8.mat');
plot(Error_valid,'-m','LineWidth',2);
h = legend('eta = 0.01','eta = 0.05','eta = 0.1','eta = 0.5','eta = 0.8',epoch);
     set(h,'Interpreter','none')