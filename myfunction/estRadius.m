function [r, minError] = estRadius(BI)
FBI = abs(fftshift(fft2(BI)));                                    %求幅度谱平移
FBIlog = log(FBI);                                                %并取对数
Y = CRadon(FBIlog);                                               %进行radon-c变换
figure, plot(Y),title('Radon-C Curve');

% 拟合前的准备
L = size(BI,1);
Lmin= 1;
Lmax= 80; 
step=0.1;
lambdas = Lmin*pi/L:step*pi/L:Lmax*pi/L;                          % lambda的候选值
erros=[];                                                         % 存储不同lambda下的拟合误差

x = [1:length(Y)-1]';                                             % 拟合的x变量
Y = Y(2:end);                                                     % 拟合的Y变量，去掉第一个是为了避免 0/0 

pnew = [10 -0.1 -0.1 6 1 round(length(x)/3)];                     % 设定拟合参数的初始值
LB = [0 -Inf -Inf 0.1 0.1 round(length(x)/10) ];                  % 拟合参数上下界限制
UB = [+Inf 0  0 +Inf +Inf length(x)];
options = optimset('display','off');

% 开始遍历lambda,求解拟合参数和对应的拟合误差
for lambda = lambdas
 
    fitfF = @(p, x) fitfunc([p(6) p(1:3)],x) + p(4).*log(1+p(5).*abs(2*besselj(1,lambda.*x)./(lambda.*x)));
    p = pnew;
    [pnew, res] = lsqcurvefit(fitfF, p, x, Y, LB, UB, options);   % 拟合
    erros = [erros res];
    
end

figure, plot(lambdas,erros), xlabel('\lambda'), title('errors : choosing the lambda which minimize the errors');

% 求解模糊核半径
[minError, indice] = min(erros);
lambda=lambdas(indice);
r = lambda*L/(2*pi);
minError = minError / L;


