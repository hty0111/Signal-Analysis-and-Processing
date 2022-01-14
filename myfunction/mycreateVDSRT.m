
function mycreateVDSRT(imds,scaleFactors,upsampledDirName,residualDirName)

if ~isfolder(residualDirName)
    mkdir(residualDirName);
end
    
if ~isfolder(upsampledDirName)
    mkdir(upsampledDirName);
end

while hasdata(imds)
    % Use only the luminance component for training
    [I,info] = read(imds);
    [~,fileName,~] = fileparts(info.Filename);
    
    tempI = im2double(I);
    tempI_R = tempI(:,:,1);                                     %拆分为R、G、B三个矩阵
    tempI_G = tempI(:,:,2);
    tempI_B = tempI(:,:,3);
    tempI = {tempI_R,tempI_G,tempI_B};
    [mrow,ncol]=size(tempI{1,1});
    
    fftI_R=fftshift(myfft2(tempI{1,1}));                                   %对图像数据进行二维傅里叶变换,并利用fftshift函数进行频谱搬移
    fftI_G=fftshift(myfft2(tempI{1,2})); 
    fftI_B=fftshift(myfft2(tempI{1,3}));
    fftI = cat(3,fftI_R,fftI_G,fftI_B);
    
    I = rgb2ycbcr(I);
    Y = I(:,:,1);
    I = im2double(Y);
    % Randomly apply one value from scaleFactor
    r = rand()*4+1;
    h1=fspecial('disk',r);                                                      % 模糊核半径为3
    [m,n]=size(h1);
    h2=padarray(h1,[mrow-m,ncol-n],'post');                                     % 填充后的模糊核
    h3=fftshift(myfft2(h2));                                                    % 模糊核频域
    fftBI=fftI.*h3;                                                         % 频域乘积
    blurImage_R = real(myifft2(ifftshift(fftBI(:,:,1))));
    blurImage_G = real(myifft2(ifftshift(fftBI(:,:,2))));
    blurImage_B = real(myifft2(ifftshift(fftBI(:,:,3))));
    blurImage = cat(3,blurImage_R,blurImage_G,blurImage_B);
    
    upsampledImage = rgb2ycbcr(blurImage);
    upsampledImage = upsampledImage(:,:,1);
    
    residualImage = I-upsampledImage;
    
    save([residualDirName filesep fileName '.mat'],'residualImage');
    save([upsampledDirName filesep fileName '.mat'],'upsampledImage');
    
end

end