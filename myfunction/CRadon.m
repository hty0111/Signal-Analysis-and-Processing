function [P] = CRadon(I)

[m,n] = size(I);
med_m = round(m/2);
med_n = round(n/2);
rho = zeros(m,n);
theta = zeros(m,n);
for i=1:m
    for j=1:n
            [theta(i,j),rho(i,j)] = cart2pol(i-med_m,j-med_n);
    end
end

% 设置step = 2;
P = zeros((round(rho(1,med_n))-1)*2,1);
num = zeros((round(rho(1,med_n))-1)*2,1)+1.0e-6;
for i=1:m
     for j=1:n
          if(floor(rho(i,j)*2+1)<=(round(rho(1,med_n))-1)*2)
               index = floor(rho(i,j)*2)+1;
               P(index) = P(index)+I(i,j);
               num(index) = num(index)+1;
          end
      end
end
P = P./num;
for i=1:(round(rho(1,med_n))-1)*2
    if(P(i)==0)
        P(i) = P(i-1);
    end
end

% 因为step =2,消去step的作用
P = P(1:2:end);

