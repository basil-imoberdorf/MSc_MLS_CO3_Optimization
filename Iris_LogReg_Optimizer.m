clear 
close all
warning('off', 'all');

% %%%%%%%%%%%%%% Iris data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data_Vault = jsondecode(fileread('iris_data.txt'));
Data = Data_Vault.data;
Target = Data_Vault.target;
Labels = Data_Vault.feature_names;

picker = 51:100;
picker2 = 101:150;

versi = Data(picker,3);
virgi = Data(picker2,3);

avg = mean([versi;virgi]);
versi = (versi - avg);
virgi = (virgi - avg);

Target_versi = Target(picker);
Target_virgi = Target(picker2);

x_1 = versi;
y_1 = Target_versi-1;
x_2 = virgi;
y_2 = Target_virgi-1;

X_PetLen = [x_1;x_2];
true_label = [y_1;y_2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% intital guess for parameters
% a_0 = -20;
% b_0 = 20;

a_0 = -7;
b_0 = -1;

% a_0 = -25;
% b_0 = -1;

% a_0 = 0;
% b_0 = 26;

% simuatlion parameters and pre-allocation of history variables
MakeMovie = 0;  % set to one to make a video form the frames
ObjFun = 'SSE'; % available are 'SSE' and 'LGL', sum of squares and log loss (binary cross-entropy)
LogFun =  '-(ax + m)'  ; %   '-(ax + m)'     or    '-(a(x - b))'
iter = 200;
Plot_Rate = 1; % defines after how many iteration the plots are updated
learning_rate = 0.1*ones(iter,1);
learning_rate = 2./(1 + exp(-0.01*((1:iter) - iter/2))) + 0.5;
learn_const = 2*ones(iter,1);
learn_const = 2./(1 + exp(0.15*((1:iter) - iter/2)));

% rng(5) % Converges for SSE ax-b with: 1,
% figure('Color','w')

% multi = 5*ones(iter,1);
% multi = 5./(1 + exp(0.01*((1:iter) - iter/1.5)));
% MaxTemp = 0.00000000000000000000000001;
% % Temp = MaxTemp*ones(iter,1);
% Temp = MaxTemp./(1 + exp(0.00001*((1:iter) - iter/2)));

% plot(1:iter,Temp,'LineWidth',2)
% hold on
% 
% Temp = MaxTemp./(1 + exp(0.1*((1:iter) - iter/2)));
% plot(1:iter,Temp,'LineWidth',2)
% 
% Temp = MaxTemp./(1 + exp(0.01*((1:iter) - iter/2)));
% plot(1:iter,Temp,'LineWidth',2)
% legend({'\lambda = 1','\lambda = 0.1','\lambda = 0.01'},'FontSize',20,'FontWeight','bold')
% xlabel('Iterations - i','FontSize',20)
% ylabel('f(i)','FontSize',24)
% ylim([0,MaxTemp])
% grid on

Loss = zeros(iter+1,1);
a = zeros(iter+1,1);
b = zeros(iter+1,1);
a(1) = a_0;
b(1) = b_0;
Loss(1) = ObjectiveFun(X_PetLen,true_label,a_0,b_0,ObjFun,LogFun);

% Grid Size
A_min = -12;
A_max = 12;
B_min = -5;
B_max = 5;

A_min = -300;
A_max = 300;
B_min = -300;
B_max = 300;

% A_min = -2000;
% A_max = 2000;
% B_min = -2000;
% B_max = 2000;

% A_min = -200000;
% A_max = 300000;
% B_min = -200000;
% B_max = 500000;
% A_min = -20;
% A_max = 15;
% B_min = -5;
% B_max = 15;

A_min = -30;
A_max = 30;
B_min = -30;
B_max = 30;

% A_min = -12;
% A_max = 12;
% B_min = -12;
% B_max = 12;

% % Zoom on Minima of Sum of Squares Error Loss when -(ax + m)
% A_min = 9.6;
% A_max = 9.8;
% B_min = 0.38;
% B_max = 0.42;
% B_min = -0.040;    % '-(a(x - b))'
% B_max = -0.043;    % '-(a(x - b))'
% % Zoom on Minima of Binary Cross-Entropy Loss when -(ax + m)
% A_min = 8.8;
% A_max = 9.2;
% B_min = 0.37;
% B_max = 0.41;
% B_min = -0.040;  % '-(a(x - b))'
% B_max = -0.043;  % '-(a(x - b))'

min_x = a_0;
min_y = b_0;
min_z = Loss(1);

runner = 0; 
xx = linspace(-4,4,300)'; % x-values to plot the updated logistic regression curve after each iteration
[paraH,lossH,modelH,teXt,ax,ay,az] = createCanvas(X_PetLen,true_label,[A_min,A_max,B_min,B_max],[a_0,b_0],ObjFun,LogFun);  % create the figure and returns the handles for each axe
pause(2)
Frame_Collection = struct('cdata', cell(1,iter/Plot_Rate + 1), 'colormap', cell(1,iter/Plot_Rate + 1)); %preallocate the array for the video frames
%%%%%%%% Optimization loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:iter
     %%%%%% Plotting &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if mod(i,Plot_Rate)==0
        paraH.XData = a(1:i);
         paraH.YData = b(1:i);
        paraH.ZData = Loss(1:i);
        paraH.CData = hot(i);  % COLORMAP
        lossH.XData = 1:i;
        lossH.YData = Loss(1:i);
        switch LogFun
            case '-(ax + m)'
                be = b(i)/a(i);
                if be < 0
                    teXt.String = ['$\hat{y} = \frac{1}{1 + e^{', sprintf('%.3f(x%.3f)', -a(i), be), '}}$'];
                else
                    teXt.String = ['$\hat{y} = \frac{1}{1 + e^{', sprintf('%.3f(x+%.3f)', -a(i), be), '}}$'];
                end
                modelH.YData = 1./(1 + exp(-(a(i).*xx + b(i))));
            case '-(a(x - b))'
                if b(i) < 0
                    teXt.String = ['$\hat{y} = \frac{1}{1 + e^{', sprintf('%.3f(x%.3f)', a(i), b(i)), '}}$'];
                else
                    teXt.String = ['$\hat{y} = \frac{1}{1 + e^{', sprintf('%.3f(x+%.3f)', a(i), b(i)), '}}$'];
                end
                modelH.YData = 1./(1 + exp(-a(i).*(xx - b(i))));
        end
        pause(0)
        % Capture Frames to make a movie
        if MakeMovie == 1
            runner = runner + 1;
            Frame_Collection(runner) = getframe(gcf);
        end
        ay.XLim = [1, iter];
    end
    %%%%%%%%%%%%% Simualted Annealing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % Perturb parameters with gaussian noise
    % a_new = a(i) + multi(i)*randn();
    % b_new = b(i) + multi(i)*randn();
    % 
    % % Evaluate objective function 
    % Loss(i+1) = ObjectiveFun(X_PetLen,true_label,a_new,b_new,ObjFun,LogFun);   % get loss
    % 
    % 
    % if Loss(i+1) < min_z
    %     min_x = a_new;
    %     min_y = b_new;
    %     min_z = Loss(i+1);
    % end
    % 
    % % Apply Simulated annealing
    % if Loss(i+1) <= Loss(i)
    %     a(i+1) = a_new;
    %     b(i+1) = b_new;
    % % elseif exp(-(Loss(i+1) - Loss(i))/Temp(i)) >= rand() % if SSE_Inter is larger we might accept it still with a probability defined by the temperature
    % %     a(i+1) = a_new;
    % %     b(i+1) = b_new;
    % else % if no improvement was made or the random sample was rejected, overwrite position i+1 with the valeus from i, the current prositon, to pick next round another position starting form the current one.
    %     a(i+1) = a(i);
    %     b(i+1) = b(i);
    %     Loss(i+1) = Loss(i); 
    % 
    %     % a(i+1) = min_x;
    %     % b(i+1) = min_y;
    % end
    %%%%%%%%%%%%% Gradient Descent  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calcualte derivatves
    [~,y_pred] = ObjectiveFun(X_PetLen,true_label,a(i),b(i),ObjFun,LogFun);
    switch ObjFun
        case 'LGL'
            % dLossdy = (y_pred - true_label)./(y_pred.*(1 - y_pred));  % same but "simplified"
            dLossdy = (1-true_label)./(1-y_pred) - true_label./y_pred;
        case 'SSE'
            dLossdy = y_pred - true_label;
    end
    switch LogFun
        case '-(ax + m)'
            % dyda = X_PetLen.*exp(-X_PetLen.*a(i) - b(i))./(1 + exp(-X_PetLen.*a(i) - b(i))).^2;
            % dydb = exp(X_PetLen.*a(i) + b(i))./(exp(X_PetLen.*a(i) + b(i)) + 1).^2;
            dyda = y_pred.*(1-y_pred).*X_PetLen;
            dydb = y_pred.*(1-y_pred);

            dLossda = sum(dLossdy .* dyda);
            dLossdb = sum(dLossdy .* dydb);

            % rescale gradients
            % learning_rate(i) = learn_const(i)/sqrt(dLossda^2 + dLossdb^2);
        case '-(a(x - b))'
            % dyda = -(b(i) - X_PetLen).*exp(-a(i).*(X_PetLen - b(i)))./(exp(-a(i).*(X_PetLen - b(i))) + 1).^2;
            % dydb = -(a(i).*exp(-a(i).*(X_PetLen - b(i))))./(exp(-a(i).*(X_PetLen - b(i))) + 1).^2;
            dyda = y_pred.*(1-y_pred).*(X_PetLen - b(i));
            dydb = y_pred.*(1-y_pred).*-a(i);

            dLossda = sum(dLossdy .* dyda);
            dLossdb = sum(dLossdy .* dydb);

            % rescale gradients
            learning_rate(i) = learn_const(i)/sqrt(dLossda^2 + dLossdb^2);
    end
    Loss(i+1) = ObjectiveFun(X_PetLen, true_label, a(i), b(i), ObjFun, LogFun);  

    a(i+1) = a(i) - learning_rate(i)*dLossda;  % apply gradient descent
    b(i+1) = b(i) - learning_rate(i)*dLossdb;  % apply gradient descent

    if Loss(i+1) < min_z
        min_x = a(i+1);
        min_y = b(i+1);
        min_z = Loss(i+1);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
ay.XLim = [1, iter];

disp(fprintf('Optimal value for slope = %.6f',min_x))
disp(fprintf('Optimal value for intercept = %.6f',min_y))

min_x
min_y

if MakeMovie == 1
    Frame_Collection(runner+1) = getframe(gcf);
    makeMovie('SA_NoConver_2000_iter__5_multi',Frame_Collection,60) % 'GD_Stuck_200iter_n_04
end

% for i = 1:240
%     ax.View(1) = ax.View(1) - 0.4;
%     ax.View(2) = ax.View(2) - 0.2;
%     pause(1/60)
% end

% py_raw = [1.4 1.5 1.5 1.3 1.5 1.3 1.6 1. 1.3 1.4 1. 1.5 1. 1.4 1.3 1.4 1.5 1. 1.5 1.1 1.8 1.3 1.5 1.2 1.3 1.4 1.4 1.7 1.5 1. 1.1 1. 1.2 1.6 1.5 1.6 1.5 1.3 1.3 1.3 1.2 1.4 1.2 1. 1.3 1.2 1.3 1.3 1.1 1.3 2.5 1.9 2.1 1.8 2.2 2.1 1.7 1.8 1.8 2.5 2. 1.9 2.1 2. 2.4 2.3 1.8 2.2 2.3 1.5 2.3 2. 2. 1.8 2.1 1.8 1.8 1.8 2.1 1.6 1.9 2. 2.2 1.5 1.4 2.3 2.4 1.8 1.8 2.1 2.4 2.3 1.9 2.3 2.5 2.3 1.9 2. 2.3 1.8];
% py = [-0.206 -0.406 -0.006 -0.906 -0.306 -0.406 -0.206 -1.606 -0.306 -1.006 -1.406 -0.706 -0.906 -0.206 -1.306 -0.506 -0.406 -0.806 -0.406 -1.006 -0.106 -0.906 -0.006 -0.206 -0.606 -0.506 -0.106  0.094 -0.406 -1.406 -1.106 -1.206 -1.006  0.194 -0.406 -0.406 -0.206 -0.506 -0.806 -0.906 -0.506 -0.306 -0.906 -1.606 -0.706 -0.706 -0.706 -0.606 -1.906 -0.806 1.094  0.194  0.994  0.694  0.894  1.694 -0.406  1.394  0.894  1.194 0.194  0.394  0.594  0.094  0.194  0.394  0.594  1.794  1.994  0.094 0.794 -0.006  1.794 -0.006  0.794  1.094 -0.106 -0.006  0.694  0.894 1.194  1.494  0.694  0.194  0.694  1.194  0.694  0.594 -0.106  0.494 0.694  0.194  0.194  0.994  0.794  0.294  0.094  0.294  0.494  0.194];
warning('on', 'all');
%%%%%%%%%%%%%%%%%%%%%%  Canvas Creation Function  %%%%%%%%%%%%%%%%%%%%%%%%%
function [paraH,lossH,modelH,teXt,ax,ay,az] = createCanvas(x_predictors,true_label,limits,initial,ObjFun,LogFun)
% create figure to plot the stuff
fig1 = figure('Position',[300,50,2100,1200],'Color','w');
ax = axes(fig1,'Position',[0.47,0.06,0.52,0.92],'NextPlot','add');
xlabel(ax,'Slope - a','FontSize',24,'FontWeight','bold')


res = 300;
[A,B] = meshgrid(linspace(limits(1),limits(2),res),linspace(limits(3),limits(4),res));
Loss = zeros(res);

for i = 1:res
    for j = 1:res
        Loss(i,j) = ObjectiveFun(x_predictors,true_label,A(i,j),B(i,j),ObjFun,LogFun);
    end
end

% surf(ax,A,B,Loss,'EdgeAlpha',0.0,'FaceAlpha',0.9)
surf(ax,A,B,Loss,'EdgeAlpha',0.0,'FaceAlpha',0.3)
contour(ax,A,B,Loss,250)
colormap(ax,'jet')

%paraH = scatter(ax,initial(1),initial(2),300,'m','Marker','+','LineWidth',2); % scatter plot for parameters on loss landscape
InitialLoss = ObjectiveFun(x_predictors,true_label,initial(1),initial(2),ObjFun,LogFun);
paraH = scatter3(ax,initial(1),initial(2),InitialLoss+10,300,'w','Marker','+','LineWidth',2); % scatter plot for parameters on loss landscape


% ax.View = [-85, 51];
ax.View = [180, 90];
ax.Color = [0.6,0.6,0.6];
% ax.ColorScale = 'log';
ax.ZScale = 'log';

%%%%% Plot the loss evolution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ay = axes(fig1,'Position',[0.04,0.56,0.36,0.42],'NextPlot','add');
xlabel('Iteration','FontSize',24,'FontWeight','bold')
ylabel('Loss','FontSize',24,'FontWeight','bold')

lossH = plot(ay,1,InitialLoss,'b-','LineWidth',3);  % plot for evolution of loss 
% ay.YScale = 'log';

%%%%% Plot the logistic regression curve %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
az = axes(fig1,'Position',[0.04,0.06,0.36,0.42],'NextPlot','add');
xlabel('Sepal Length [cm] (mean centerd)','FontSize',24,'FontWeight','bold')
ylabel('Probability ','FontSize',24,'FontWeight','bold')
miN = -4;
maX = 4;
scatter(az, x_predictors(1:50)  , true_label(1:50)  , 800, 'LineWidth',2);  % scatter plot for VERSICOLOR data points 
scatter(az, x_predictors(51:end), true_label(51:end), 800, 'LineWidth',2);  % scatter plot for VIRIGNICA data points 
plot([miN,maX],[0,0],'-k','LineWidth',1); % indicate y = 0
plot([miN,maX],[1,1],'-k','LineWidth',1); % indicate y = 1

% x = linspace(B_min,B_max,300);
x = linspace(miN,maX,300)';

% Adapt the optimal parameters according to the used objective function and type of logistic function
switch ObjFun  % optimal values for when y_pred = 1/(1 + exp( ax + b ))
    case 'LGL'
        a_opt = -9.002;
        b_opt = -0.3829;
    case 'SSE'
        a_opt = -9.695;
        b_opt = -0.4019;
end
y_best_fit = 1./(1 + exp(a_opt*x + b_opt));
plot(x,y_best_fit,'Color',[1,0,0.35],'LineWidth',1); % plot current logistic fit
text(az,0.3,0.2,['$\hat{y}_{opt} = \frac{1}{1 + e^{', sprintf('%.3f(x+%.3f)', a_opt, b_opt/a_opt), '}}$'], 'Interpreter', 'latex', 'FontSize', 32,'Color',[0.8,0,0.15]);                % optimal curve fit
switch LogFun
    case '-(ax + m)' % negate/invert both parameters
        a_opt = -a_opt;
        b_opt = -b_opt;
        ylabel(ax,'Intercept - m','FontSize',24,'FontWeight','bold')
        y = 1./(1 + exp( -initial(1).*x - initial(2)));
        modelH = plot(x,y,'Color',[0.35,0,1],'LineWidth',4); % plot logistic fit
        if initial(2)/initial(1)==Inf || initial(2)/initial(1)==-Inf
            teXt = text(az,-3.5,0.8,['$\hat{y} = \frac{1}{1 + e^{', sprintf('%.3f(x-%.3f)', -initial(1), 0), '}}$'], 'Interpreter', 'latex', 'FontSize', 32,'Color',[0.25,0,0.9]); % current curve fit
        else
            teXt = text(az,-3.5,0.8,['$\hat{y} = \frac{1}{1 + e^{', sprintf('%.3f(x+%.3f)', -initial(1), initial(2)/initial(1)), '}}$'], 'Interpreter', 'latex', 'FontSize', 32,'Color',[0.25,0,0.9]); % current curve fit
        end
    case '-(a(x - b))'  % invert a, and b = -m/a
        b_opt = -b_opt/a_opt;
        a_opt = -a_opt;
        ylabel(ax,'Intercept - b','FontSize',24,'FontWeight','bold')
        y = 1./(1 + exp( -initial(1).*x + initial(2)/initial(1)));
        modelH = plot(x,y,'Color',[0.35,0,1],'LineWidth',4); % plot logistic fit
        teXt = text(az,-3.5,0.8,['$\hat{y} = \frac{1}{1 + e^{', sprintf('%.3f(x+%.3f)', -initial(1), initial(2)), '}}$'], 'Interpreter', 'latex', 'FontSize', 32,'Color',[0.25,0,0.9]); % current curve fit
end
% scatter3(ax, a_opt, b_opt, ObjectiveFun(x_predictors,true_label,-160,-160,ObjFun,LogFun), 100,'y', 'LineWidth',3,'Marker','.','MarkerEdgeColor','m');  % indicated global optimum with scatter point
scatter3(ax, a_opt, b_opt, 5000, 100,'y', 'LineWidth',3,'Marker','.','MarkerEdgeColor','m');  % indicated global optimum with scatter point


% Formating
az.YLim = [-0.05,1.05];
ax.XGrid = 'on';    ax.YGrid = 'on';    ax.ZGrid = 'on';  ax.Box = 'on';
ay.XGrid = 'on';    ay.YGrid = 'on';    ay.ZGrid = 'on';  ay.Box = 'on';
az.XGrid = 'on';    az.YGrid = 'on';    az.ZGrid = 'on';  az.Box = 'on';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Loss,pred_label_probability] = ObjectiveFun(X,true_label,a,b,Objective_Function_Mode,Logistic_Function_Mode)

% X = [x';ones(1,numel(x))];
% w = [a;b];
% z = w'*X;

switch Logistic_Function_Mode
    case '-(ax + m)'
        pred_label_probability = 1./(1 + exp(  -(a.*X + b) )); % get probabiltiy
    case  '-(a(x - b))'
        pred_label_probability = 1./(1 + exp( -(a.*(X - b)) )); % get probabiltiy
end

switch Objective_Function_Mode
    case 'SSE' % sum of squares
        Loss = 0.5*sum((true_label - pred_label_probability).^2);   
    case 'LGL' % log loss is binary cross-entropy loss in this case
        Loss = sum(-true_label.*log(pred_label_probability) - (1-true_label).*log(1-pred_label_probability));
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function makeMovie(Filename,Frame_Collection,Frame_Rate)
%create a .mp4 video clip based on the frames contained in the variable "Frame_Collection"

%Filename: input desired filename as string with ending .mp4
%Frame_Collection: Array containing object. Normaly generated in a loop via getframe: Frame_Collection(loop_index) = getframe(gcf)
%FrameRate: optional parameter that defines how many frames are displayed per second. Default is 60 Fps

switch nargin
    case 2
    Frame_Rate = 60;
end

Movie = VideoWriter(Filename,'MPEG-4');
Movie.FrameRate = Frame_Rate;
Movie.Quality = 100;
open(Movie)
writeVideo(Movie,Frame_Collection)
close(Movie)
end

