function [Y] = myifft2(X)

size_X=size(X);
mrow=size_X(1);
ncol=size_X(2);

G1=zeros(mrow,mrow);
G2=zeros(ncol,ncol);

for m=1:mrow
    for n=1:mrow
        G1(m,n)=exp(j*2*pi*(m-1)*(n-1)/mrow);
    end
end

for m=1:ncol
    for n=1:ncol
        G2(m,n)=exp(j*2*pi*(m-1)*(n-1)/ncol);
    end
end

Y=G1*X*G2/(mrow * ncol);

end