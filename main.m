addpath(genpath('myfunction'));
load('datasets/trainedModel/LSTMNet.mat');
% Read image
originalImage = im2double(imread('original.jpg'));                          % ��ȡͼ��
originalImage_R = originalImage(:,:,1);                                     %���ΪR��G��B��������
originalImage_G = originalImage(:,:,2);
originalImage_B = originalImage(:,:,3);
originalImages = {originalImage_R,originalImage_G,originalImage_B};         %������ϳ���ά���󣬷���ֱ�����


% Simulate image
% originalImage=zeros(200,200);
% for mr=15:70
%     for nc=20:90
%         originalImage(mr,nc)=1;
%     end
% end

% get the rows and colums
[mrow,ncol]=size(originalImages{1,1});                                              %��ȡͼ�����ݵĺ�������
fftImage_R=fftshift(myfft2(originalImages{1,1}));                                   %��ͼ�����ݽ��ж�ά����Ҷ�任,������fftshift��������Ƶ�װ���
fftImage_G=fftshift(myfft2(originalImages{1,2})); 
fftImage_B=fftshift(myfft2(originalImages{1,3})); 
figure,subplot(3,2,1),imshow(originalImage);                                        %����ԭʼͼƬ
title('Original Image');
fftImage = cat(3,fftImage_R,fftImage_G,fftImage_B);                                 %����Ҷ�任��������ϳ���ά����
subplot(3,2,2),imshow(fftImage);                                                    %����ԭʼ��Ƶ��ͼ��
title('Original Image fft');

% disk blur
r=5;
h1=fspecial('disk',r);                                                      % ģ���˰뾶Ϊ3
[m,n]=size(h1);
h2=padarray(h1,[mrow-m,ncol-n],'post');                                     % �����ģ����
h3=fftshift(myfft2(h2));                                                    % ģ����Ƶ��
fftBI=fftImage.*h3;                                                         % Ƶ��˻�
blurImage_R=real(myifft2(ifftshift(fftBI(:,:,1))));
blurImage_G=real(myifft2(ifftshift(fftBI(:,:,2))));
blurImage_B=real(myifft2(ifftshift(fftBI(:,:,3))));
blurImage=cat(3,blurImage_R,blurImage_G,blurImage_B);
save('blurImage.mat', 'blurImage');
%gaussian blur
% sigma=3;
% h1=fspecial('gaussian',sigma);                                            % ģ����
% h2=padarray(h1,[mrow-fix(sigma),ncol-fix(sigma)],'post');                 % �����ģ����
% h3=fftshift(myfft2(h2));                                                  % ģ����Ƶ��
% fftBI=fftImage.*h3;                                                       % Ƶ��˻�
% blurImage_R=real(myifft2(ifftshift(fftBI(:,:,1))));
% blurImage_G=real(myifft2(ifftshift(fftBI(:,:,2))));
% blurImage_B=real(myifft2(ifftshift(fftBI(:,:,3))));
% blurImage=cat(3,blurImage_R,blurImage_G,blurImage_B);

subplot(3,2,3),imshow(h2);
title('disk blur');
subplot(3,2,4),imshow(real(h3));
title('disk blur fft');
subplot(3,2,5),imshow(blurImage);
title('disk blur Image');
subplot(3,2,6),imshow(fftBI);
title('disk blur Image fft');

% disblur
FBI = abs(fftshift(fft2(blurImage)));                                    %�������ƽ��
FBIlog = log(FBI);                                                %��ȡ����
Y = CRadon(FBIlog);                                               %����radon-c�任
estr = double(net.activations(Y',6));
% [estr, minError] = estRadius(blurImage); % ����ģ���뾶
esth1=fspecial('disk',estr);                                                % ����ģ����
[estm,estn]=size(esth1);
esth2=padarray(esth1,[mrow-estm,ncol-estn],'post');                         % ����Ĺ���ģ����
esth3=fftshift(myfft2(esth2));                                              % ����ģ���˵�Ƶ��
restorefft=mydeconvwnr(fftBI,esth3, 0.003);                                   % ά���˲�
restoration_R=real(myifft2(ifftshift(restorefft(:,:,1))));                  % ���˲����ɵ�ͼ��
restoration_G=real(myifft2(ifftshift(restorefft(:,:,2))));
restoration_B=real(myifft2(ifftshift(restorefft(:,:,3))));
restoration=cat(3,restoration_R,restoration_G,restoration_B);
% restoration = restorefft;

figure,subplot(3,2,1),imshow(originalImage);
title('Original Image');
subplot(3,2,2),imshow(fftImage);
title('Original Image fft');
subplot(3,2,3),imshow(esth2);
title('estimated disk blur');
subplot(3,2,4),imshow(real(esth3));
title('estimated disk blur fft');
subplot(3,2,5),imshow(restoration);
title('restored Image');
subplot(3,2,6),imshow(restorefft);
title('restored Image fft');
figure(5);
imshow(originalImage);
title('original');
figure(6);
imshow(blurImage);
title('blur');
figure(7);
imshow(restoration);
title('restored');

PEAKSNR = psnr(restoration, originalImage);                                  % ����PSNR
SSIM = ssim(restoration, originalImage);