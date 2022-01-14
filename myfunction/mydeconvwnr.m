function [result] = mydeconvwnr(blurImage,PSF,nsr)
buf=(abs(PSF)).^2;
SF=(abs(blurImage)).^2;
Gf=1./PSF.*buf./(buf+nsr);
result=Gf.*blurImage;
end

