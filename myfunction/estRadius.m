function [r, minError] = estRadius(BI)
FBI = abs(fftshift(fft2(BI)));                                    %�������ƽ��
FBIlog = log(FBI);                                                %��ȡ����
Y = CRadon(FBIlog);                                               %����radon-c�任
figure, plot(Y),title('Radon-C Curve');

% ���ǰ��׼��
L = size(BI,1);
Lmin= 1;
Lmax= 80; 
step=0.1;
lambdas = Lmin*pi/L:step*pi/L:Lmax*pi/L;                          % lambda�ĺ�ѡֵ
erros=[];                                                         % �洢��ͬlambda�µ�������

x = [1:length(Y)-1]';                                             % ��ϵ�x����
Y = Y(2:end);                                                     % ��ϵ�Y������ȥ����һ����Ϊ�˱��� 0/0 

pnew = [10 -0.1 -0.1 6 1 round(length(x)/3)];                     % �趨��ϲ����ĳ�ʼֵ
LB = [0 -Inf -Inf 0.1 0.1 round(length(x)/10) ];                  % ��ϲ������½�����
UB = [+Inf 0  0 +Inf +Inf length(x)];
options = optimset('display','off');

% ��ʼ����lambda,�����ϲ����Ͷ�Ӧ��������
for lambda = lambdas
 
    fitfF = @(p, x) fitfunc([p(6) p(1:3)],x) + p(4).*log(1+p(5).*abs(2*besselj(1,lambda.*x)./(lambda.*x)));
    p = pnew;
    [pnew, res] = lsqcurvefit(fitfF, p, x, Y, LB, UB, options);   % ���
    erros = [erros res];
    
end

figure, plot(lambdas,erros), xlabel('\lambda'), title('errors : choosing the lambda which minimize the errors');

% ���ģ���˰뾶
[minError, indice] = min(erros);
lambda=lambdas(indice);
r = lambda*L/(2*pi);
minError = minError / L;


