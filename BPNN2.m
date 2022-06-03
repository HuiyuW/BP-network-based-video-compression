function [image_rec,bit] = BPNN2(image,K,N)

row=256; % 设置所有进行压缩的图像的原始大小都为256x256
col=256;

[rows, columns, numberOfColorChannels] = size(image);
image_rec = zeros(size(image));
bit_length = zeros(1,numberOfColorChannels);

%% Y
imagere1 = imresize(image(:,:,1),[row,col]); % Resize image

P1=block_divide(imagere1,K); % 调用自定义函数block_divide将图像进行划分，形成K^2 x N大小的矩阵

P1=double(P1)/255; % 对每一个像素点进行归一化处理
 
net=feedforwardnet(N,'trainlm'); % Feedforward neural network
T1=P1;
net.trainParam.goal=0.001; % 设置BP网络训练参数
net.trainParam.epochs=500;
% tic % Start stopwatch timer
net=train(net,P1,T1); % Train neural network
% toc % Read elapsed time from stopwatch
 

com1.lw=net.lw{2};
com1.b=net.b{2};
[~,len1]=size(P1); % 训练样本的个数
com1.d=zeros(N,len1);
for j=1:len1
    com1.d(:,j)=tansig(net.iw{1}*P1(:,j)+net.b{1}); % Hyperbolic tangent sigmoid transfer function
end
minlw= min(com1.lw(:));
maxlw= max(com1.lw(:));
com1.lw=(com1.lw-minlw)/(maxlw-minlw);
minb= min(com1.b(:));
maxb= max(com1.b(:));
com1.b=(com1.b-minb)/(maxb-minb);
maxd=max(com1.d(:));
mind=min(com1.d(:));
com1.d=(com1.d-mind)/(maxd-mind);
 
com1.lw=uint8(com1.lw*63);
com1.b=uint8(com1.b*63);
com1.d=uint8(com1.d*63);
 
save comp1 com1 minlw maxlw minb maxb maxd mind


% bp_imageRecon.m
 
 
% I=imread('lena.tif'); % 重新载入原始图片，用作对比
% I = imresize(I,[row,col]);
load comp1
com1.lw=double(com1.lw)/63;
com1.b=double(com1.b)/63;
com1.d=double(com1.d)/63;
com1.lw=com1.lw*(maxlw-minlw)+minlw;
com1.b=com1.b*(maxb-minb)+minb;
com1.d=com1.d*(maxd-mind)+mind;
 

for i=1:4096
   Y1(:,i)=com1.lw*(com1.d(:,i)) +com1.b;
end
 

Y1=uint8(Y1*255);
 

I1=re_divide(Y1,col,4); % 将重建后的图片存储在I1变量中
I2 = imresize(I1, [288,352]);
image_rec(:,:,1) = I2;

% fprintf('PSNR :\n  ');
% psnr=10*log10(255^2*row*col/sum(sum((image(:,:,1)-I2).^2)));
% disp(psnr)
a=dir();
for m=1:length(a)
   if (strcmp(a(m).name,'comp1.mat')==1) 
       si1=a(m).bytes;
       break;
   end
end
bit_length(1) = si1;
% fprintf('rate: \n  ');
% rate=double(si)/(256*256);
% disp(rate) % Display value of variable
% figure(1) % Create figure window
% imshow(image(:,:,1)) % Display image
% title('原始图像');
% figure(2)
% imshow(I2)
% title('重建图像');  
%% Cb
imagere2 = imresize(image(:,:,2),[row,col]); % Resize image

P2=block_divide(imagere2,K); % 调用自定义函数block_divide将图像进行划分，形成K^2 x N大小的矩阵

P2=double(P2)/255; % 对每一个像素点进行归一化处理
 
net=feedforwardnet(N,'trainlm'); % Feedforward neural network
T2=P2;
net.trainParam.goal=0.001; % 设置BP网络训练参数
net.trainParam.epochs=500;
% tic % Start stopwatch timer
net=train(net,P2,T2); % Train neural network
% toc % Read elapsed time from stopwatch
 

com2.lw=net.lw{2};
com2.b=net.b{2};
[~,len2]=size(P2); % 训练样本的个数
com2.d=zeros(N,len2);
for j=1:len2
    com2.d(:,j)=tansig(net.iw{1}*P2(:,j)+net.b{1}); % Hyperbolic tangent sigmoid transfer function
end
minlw= min(com2.lw(:));
maxlw= max(com2.lw(:));
com2.lw=(com2.lw-minlw)/(maxlw-minlw);
minb= min(com2.b(:));
maxb= max(com2.b(:));
com2.b=(com2.b-minb)/(maxb-minb);
maxd=max(com2.d(:));
mind=min(com2.d(:));
com2.d=(com2.d-mind)/(maxd-mind);
 
com2.lw=uint8(com2.lw*63);
com2.b=uint8(com2.b*63);
com2.d=uint8(com2.d*63);
 
save comp2 com2 minlw maxlw minb maxb maxd mind


% bp_imageRecon.m
 
 
% I=imread('lena.tif'); % 重新载入原始图片，用作对比
% I = imresize(I,[row,col]);
load comp2
com2.lw=double(com2.lw)/63;
com2.b=double(com2.b)/63;
com2.d=double(com2.d)/63;
com2.lw=com2.lw*(maxlw-minlw)+minlw;
com2.b=com2.b*(maxb-minb)+minb;
com2.d=com2.d*(maxd-mind)+mind;
 

for i=1:4096
   Y2(:,i)=com2.lw*(com2.d(:,i)) +com2.b;
end
 

Y2=uint8(Y2*255);
 

I1=re_divide(Y2,col,4); % 将重建后的图片存储在I1变量中
I2 = imresize(I1, [288,352]);
image_rec(:,:,2) = I2;

% fprintf('PSNR :\n  ');
% psnr=10*log10(255^2*row*col/sum(sum((image(:,:,2)-I2).^2)));
% disp(psnr)
a=dir();
for m=1:length(a)
   if (strcmp(a(m).name,'comp1.mat')==1) 
       si2=a(m).bytes;
       break;
   end
end
bit_length(2) = si2;


%% Cr
imagere3 = imresize(image(:,:,3),[row,col]); % Resize image

P3=block_divide(imagere3,K); % 调用自定义函数block_divide将图像进行划分，形成K^2 x N大小的矩阵

P3=double(P3)/255; % 对每一个像素点进行归一化处理
 
net=feedforwardnet(N,'trainlm'); % Feedforward neural network
T3=P3;
net.trainParam.goal=0.001; % 设置BP网络训练参数
net.trainParam.epochs=500;
% tic % Start stopwatch timer
net=train(net,P3,T3); % Train neural network
% toc % Read elapsed time from stopwatch
 

com3.lw=net.lw{2};
com3.b=net.b{2};
[~,len2]=size(P3); % 训练样本的个数
com3.d=zeros(N,len2);
for j=1:len2
    com3.d(:,j)=tansig(net.iw{1}*P3(:,j)+net.b{1}); % Hyperbolic tangent sigmoid transfer function
end
minlw= min(com3.lw(:));
maxlw= max(com3.lw(:));
com3.lw=(com3.lw-minlw)/(maxlw-minlw);
minb= min(com3.b(:));
maxb= max(com3.b(:));
com3.b=(com3.b-minb)/(maxb-minb);
maxd=max(com3.d(:));
mind=min(com3.d(:));
com3.d=(com3.d-mind)/(maxd-mind);
 
com3.lw=uint8(com3.lw*63);
com3.b=uint8(com3.b*63);
com3.d=uint8(com3.d*63);
 
save comp3 com3 minlw maxlw minb maxb maxd mind


% bp_imageRecon.m
 
 
% I=imread('lena.tif'); % 重新载入原始图片，用作对比
% I = imresize(I,[row,col]);
load comp3
com3.lw=double(com3.lw)/63;
com3.b=double(com3.b)/63;
com3.d=double(com3.d)/63;
com3.lw=com3.lw*(maxlw-minlw)+minlw;
com3.b=com3.b*(maxb-minb)+minb;
com3.d=com3.d*(maxd-mind)+mind;
 

for i=1:4096
   Y3(:,i)=com3.lw*(com3.d(:,i)) +com3.b;
end
 

Y3=uint8(Y3*255);
 

I1=re_divide(Y3,col,4); % 将重建后的图片存储在I1变量中
I2 = imresize(I1, [288,352]);
image_rec(:,:,3) = I2;

% fprintf('PSNR :\n  ');
% psnr=10*log10(255^2*row*col/sum(sum((image(:,:,3)-I2).^2)));
% disp(psnr)
a=dir();
for m=1:length(a)
   if (strcmp(a(m).name,'comp1.mat')==1) 
       si3=a(m).bytes;
       break;
   end
end
bit_length(3) = si3;
%%


bit = sum(bit_length);


end