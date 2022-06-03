function [final,data_len] = JPEG(image, qScale)

image_YCbCr = image; 
BlockSize = 8;   %la taille de chaque bloque 
[rows1, columns1, numberOfColorChannels] = size(image_YCbCr);

e1=floor(rows1/BlockSize);
e2=floor(columns1/BlockSize);

 h1 = e1*BlockSize;
 h2 = e2*BlockSize;
 
imNTU = imresize(image_YCbCr,[h1 h2]) ;

info = imfinfo('foreman0020.bmp');
ImageSize = info.FileSize;

a = size(imNTU);                  
w = a(1); h = a(2);
Y =  imNTU(:,:,1);
Cb = imNTU(:,:,2);
Cr = imNTU(:,:,3);

Cb1 = Cb;
Cr1 = Cr;

I_dct = blockproc(imNTU, [8, 8], @(block_struct) DCT8x8(block_struct.data));

q_dct = blockproc(I_dct, [8, 8], @(block_struct) Quant8x8_q(block_struct.data,qScale)); 

YQ = q_dct(:,:,1);
CbQ = q_dct(:,:,2);
CrQ = q_dct(:,:,3);

%% DC differential encoding (Interlaced Scanning   Y => Cb => Cr )
% AC  ZigZag scanning
[DC,  AC_zig] = DCdif_fntest2(YQ,CbQ,CrQ,w,h); %% DC+AC

%% Huffman Encoding
HuffDC = DC_Huff_fn(DC);

HuffAC = AC_Huff(AC_zig);



data_len = length(HuffAC) + length(HuffDC);  
bit_rate = data_len/w/h;
Compression_Ratio = 24/bit_rate;
compressedsize = ImageSize/Compression_Ratio;
%% JPEG Decoder
AC_decode = AC_De_Huff(HuffAC);
DC_decode = DC_De_Huff(HuffDC);
[YY CbCb CrCr] = MatCom_fntest4(DC_decode,AC_decode,w,h,qScale);
% Cbr = four_two_zero_recovery(width,height,CbCb);
% Crr = four_two_zero_recovery(width,height,CrCr);
 Cbr = CbCb;
 Crr = CrCr;

final = cat(3,YY,Cbr,Cbr) ;

end