% Image Compression (RGB)
clc; clear;
tic;
%% Fetch
% imNTU = imread('foreman0020.bmp');  




image = cell(1,20);
 for i= 1:20
        picturespath= ['sequences/foreman20_40_RGB/foreman00',int2str(19+i),'.bmp'];
        image{i} = double(imread(picturespath));
 end
 
image_YCbCr = cell(1,20);
for i= 1:20
        image_YCbCr{i} = ictRGB2YCbCr(image{i}); 
end

% gscales = [0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5];
gscales = [ 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5];
% gscales = [ 0.4, 0.8];
num_gscale = numel(gscales);
bpp_intermean1 = zeros(1,num_gscale);
psnr_intermean1 = zeros(1,num_gscale);

bpp_intrahmean6 = zeros(1,num_gscale);
psnr_intrahmean6 = zeros(1,num_gscale);

% indexDeblocking = [1,1,26,28,32,36,35,37,38];%1,1.5
indexDeblocking = [1,26,28,32,36,35,37,38];%1,1.5
% indexDeblocking = [1,26];%1,1.5
for scaleindex = 1 : num_gscale
indexCurrent = indexDeblocking(scaleindex);   
range = -1000:4000;
num_Image = numel(image);
decoded_frame = cell(num_Image,1);        % decoded images

img_size = size(image{1});
range = -1000:4000;

bpp_inter = zeros(1,num_Image);
psnr_inter = zeros(1,num_Image);
bpp_intrah = zeros(1,num_Image);
psnr_intrah = zeros(1,num_Image);

qScale = gscales(scaleindex);

[final,data_len] = JPEG1(image_YCbCr{1},qScale);
final = deblocking_filter(final,indexCurrent);

decoded_frame{1} = final;
finalRGB = ictYCbCr2RGB(final);

bpp_intrag = data_len / (numel(image{1})/3);
bpp_intrah(1) = bpp_intrag;
psnr_intrag = calcPSNR(image{1}, finalRGB);
psnr_intrah(1) = psnr_intrag;

 % set first data of inter mode to intra.
bpp_inter(1) = bpp_intrag;
psnr_inter(1) = psnr_intrag; 


 %% Use the prediction error of the second frame to train Huffmann code.
  ref_image1 = decoded_frame{1};                   
  curr_image2 = image_YCbCr{2};
  %Only the luminance component
  motion_vectors2 = SSD(ref_image1(:,:,1), curr_image2(:,:,1));     
  rec_image2 = SSD_rec(ref_image1, motion_vectors2);             
  err_image2 = curr_image2 - rec_image2; % err_image here is 3-dim                 
  k_err2= IntraEncode(err_image2,qScale); %set err_image to zerorun string.
  pmf_mo = stats_marg(motion_vectors2(:), (1:81));% 81 according to motion vector indexing
  [BinaryTree_mo, BinCode_mo, Codelengths_mo] = buildHuffman(pmf_mo);
  pmf_err = stats_marg(k_err2,range);% according to requirement.
  [BinaryTree_err, BinCode_err, Codelengths_err] = buildHuffman(pmf_err);
  %% Inter mode in gscales 
  % image num starts from 2.
    for i = 2 : num_Image
        %set last image as referece image.
        ref_image = decoded_frame{i-1};                   
        curr_image = image_YCbCr{i};
        %Only the luminance component
        motion_vectors = SSD(ref_image(:,:,1), curr_image(:,:,1));     
        rec_image = SSD_rec(ref_image, motion_vectors);             
        err_image = curr_image - rec_image; % err_image here is 3-dim                 
        k_err= IntraEncode(err_image,qScale); %set err_image to zerorun string.                               
        
   %% Inter mode en-decode.
            
        k_e0 = k_err - (-1000-1); %set k_err larger than 0.
        bytestream_err = enc_huffman_new(k_e0, BinCode_err, Codelengths_err);
        bpp_err = (numel(bytestream_err)*8) / (numel(curr_image)/3);
        
        bytestream_mo = enc_huffman_new(motion_vectors(:), BinCode_mo, Codelengths_mo);
        bpp_mo = (numel(bytestream_mo)*8) / (numel(curr_image)/3);
        
        k_err_rec = dec_huffman_new(bytestream_err, BinaryTree_err, numel(k_err)) + (-1000-1);% remember add to original interval.
        err_rec = IntraDecode(k_err_rec, img_size, qScale);
        motion_vectors_rec = reshape(dec_huffman_new(bytestream_mo, BinaryTree_mo, numel(motion_vectors(:))), size(motion_vectors));
        % each time the rec_frame consists of pieces from ref_image and
        % err_rec , so that images will not always be the same.
        decoded_frame{i} = SSD_rec(ref_image, motion_vectors_rec) + err_rec;
        image_rec = ictYCbCr2RGB(decoded_frame{i});
        image_rec = deblocking_filter(image_rec,indexCurrent);
        
        bpp_inter(i) = bpp_err + bpp_mo;               
        psnr_inter(i) = calcPSNR(image_rec, image{i});
         
%         [finals,data_lens] = JPEG1(image_YCbCr{i}, qScale);
%         finals = deblocking_filter(finals,indexCurrent);
%         finalsRGB = ictYCbCr2RGB(finals);
%         bpp_intrah(i) = (data_lens) / (numel(image{i})/3);
%         psnr_intrah(i) = calcPSNR(finalsRGB, image{i});
       
    end
%     bpp_intrahmean6(scaleindex) = sum(bpp_intrah)/length(bpp_intrah); 
%     psnr_intrahmean6(scaleindex) = sum(psnr_intrah)/length(psnr_intrah);

    bpp_intermean1(scaleindex) = sum(bpp_inter)/length(bpp_inter);   
   psnr_intermean1(scaleindex) = sum(psnr_inter)/length(psnr_inter);
   
   fprintf('q_scaleg: %4.2f  bit-rate: %8.4f bits/pixel  PSNR: %8.4fdB\n', qScale, bpp_intermean1(scaleindex), psnr_intermean1(scaleindex));
%    fprintf('q_scaleh: %4.2f  bit-rate: %8.4f bits/pixel  PSNR: %8.4fdB\n', qScale, bpp_intrahmean6(scaleindex), psnr_intrahmean6(scaleindex));
end

%% Plot DR 
load('./chapter4.mat')
load('./chapter5.mat')
load('./Optimization26.mat')
plot(bitPerPixel_foreman, PSNR_foreman,'p-','MarkerFaceColor','r','MarkerEdgeColor','k')
hold on
plot(bpp_intermean, psnr_intermean,'p-','MarkerFaceColor','b','MarkerEdgeColor','k')
hold on
plot(bpp_intermean1, psnr_intermean1,'d-','MarkerFaceColor','g','MarkerEdgeColor','k')
hold on
plot(bpp_intermean2, psnr_intermean2,'d-','MarkerFaceColor','y','MarkerEdgeColor','k')
xlabel('bit/pixel')
ylabel('PSNR[dB]')
xlim([0.2 4])
title('D-R curve')
% legend(["Video Codec5";"Still Image Codec5";"Video Codec6";"Still Image Codec6"],'location','southeast')
legend(["Codec4";"Video Codec5";"Optimization1";"Optimization2"],'location','southeast')
grid on


toc

%% Intra mode function(encode decode)

function dst = IntraEncode(image, qScale)
%dct
I_dct = blockproc(image, [8, 8], @(block_struct) DCT8x8(block_struct.data));
%quantiztion
q_dct = blockproc(I_dct, [8, 8], @(block_struct) Quant8x8(block_struct.data,qScale)); 
%%zig_zag
I_zig_zag = blockproc(q_dct, [8, 8], @(block_struct) ZigZag8x8(block_struct.data));
%zero_run coding
I_zig_zag = reshape(I_zig_zag,1,[]);
dst = ZeroRunEnc_EoB(I_zig_zag,4000);
end

function dst = IntraDecode(image, img_size , qScale)

zerodec = ZeroRunDec_EoB(image,4000);
h_size = img_size(1);
w_size = img_size(2);
num_block = max(ceil(h_size/8),ceil(w_size/8));
zerodec = reshape(zerodec,[],3*num_block);
%%DeZigZag8x8()
dezig = blockproc(zerodec, [64, 3], @(block_struct) DeZigZag8x8(block_struct.data)); 
dezig = reshape(dezig,img_size);
% DeQuant8x8()
dequant = blockproc(dezig, [8, 8], @(block_struct) DeQuant8x8(block_struct.data,qScale)); 
% Idct 
I_dct = blockproc(dequant, [8, 8], @(block_struct) IDCT8x8(block_struct.data));
%%rgb
dst = I_dct;
end

%% DCT / IDCT

function coeff = DCT8x8(block)
    coeff(:,:,1) = dct2(block(:,:,1));
    coeff(:,:,2) = dct2(block(:,:,2));
    coeff(:,:,3) = dct2(block(:,:,3));
end

function block = IDCT8x8(coeff)
    block(:,:,1) = idct2(coeff(:,:,1));
    block(:,:,2) = idct2(coeff(:,:,2));
    block(:,:,3) = idct2(coeff(:,:,3));
end

%% Quantization

function quant = Quant8x8(dct_block, qScale)
    Luminance = qScale*[16 11 10 16 24 40 51 61;
                        12 12 14 19 26 58 60 55;
                        14 13 16 24 40 57 69 56;
                        14 17 22 29 51 87 80 62;
                        18 55 37 56 68 109 103 77;
                        24 35 55 64 81 104 113 92;
                        49 64 78 87 103 121 120 101;
                        72 92 95 98 112 100 103 99];
    
    Chrominance = qScale*[17 18 24 47 99 99 99 99;
                          18 21 26 66 99 99 99 99;
                          24 13 56 99 99 99 99 99;
                          47 66 99 99 99 99 99 99;
                          99 99 99 99 99 99 99 99;
                          99 99 99 99 99 99 99 99;
                          99 99 99 99 99 99 99 99;
                          99 99 99 99 99 99 99 99];
            
    quant(:,:,1) = round(dct_block(:,:,1)./Luminance);
    quant(:,:,2) = round(dct_block(:,:,2)./Chrominance);
    quant(:,:,3) = round(dct_block(:,:,3)./Chrominance);
end

function dct_block = DeQuant8x8(quant_block, qScale)
    Luminance = qScale*[16 11 10 16 24 40 51 61;
                        12 12 14 19 26 58 60 55;
                        14 13 16 24 40 57 69 56;
                        14 17 22 29 51 87 80 62;
                        18 55 37 56 68 109 103 77;
                        24 35 55 64 81 104 113 92;
                        49 64 78 87 103 121 120 101;
                        72 92 95 98 112 100 103 99];
    
    Chrominance = qScale*[17 18 24 47 99 99 99 99;
                          18 21 26 66 99 99 99 99;
                          24 13 56 99 99 99 99 99;
                          47 66 99 99 99 99 99 99;
                          99 99 99 99 99 99 99 99;
                          99 99 99 99 99 99 99 99;
                          99 99 99 99 99 99 99 99;
                          99 99 99 99 99 99 99 99];
            
    dct_block(:,:,1) = quant_block(:,:,1).*Luminance;
    dct_block(:,:,2) = quant_block(:,:,2).*Chrominance;
    dct_block(:,:,3) = quant_block(:,:,3).*Chrominance;
end

function quant = Quant8x8_q(dct_block, qScale)
quanTable_L = [4,6,7,8,10,13,18,31; 6,9,10,12,15,20,28,48;...
    7,10,12,14,18,23,32,55; 8,12,14,17,21,27,38,65;...
    10,15,18,21,26,33,47,80; 13,20,23,27,33,43,61,103;...
    18,28,32,38,47,61,86,146; 31,48,55,65,80,103,146,250];

quanTable_C = [17,18,24,47,99,99,99,99; 18,21,26,66,99,99,99,99;...
    24,13,56,99,99,99,99,99; 47,66,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99];

quant(:,:,1) =  round(dct_block(:,:,1) ./ (quanTable_L .* qScale));
quant(:,:,2) =  round(dct_block(:,:,2) ./ (quanTable_C .* qScale));
quant(:,:,3) =  round(dct_block(:,:,3) ./ (quanTable_C .* qScale));
end


function dct_block = DeQuant8x8_q(quant_block, qScale)

quanTable_L = [4,6,7,8,10,13,18,31; 6,9,10,12,15,20,28,48;...
    7,10,12,14,18,23,32,55; 8,12,14,17,21,27,38,65;...
    10,15,18,21,26,33,47,80; 13,20,23,27,33,43,61,103;...
    18,28,32,38,47,61,86,146; 31,48,55,65,80,103,146,250];

quanTable_C = [17,18,24,47,99,99,99,99; 18,21,26,66,99,99,99,99;...
    24,13,56,99,99,99,99,99; 47,66,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99];

dct_block(:,:,1) = round(quant_block(:,:,1) .* (quanTable_L .* qScale));
dct_block(:,:,2) = round(quant_block(:,:,2) .* (quanTable_C .* qScale));
dct_block(:,:,3) = round(quant_block(:,:,3) .* (quanTable_C .* qScale));
end

%% Zigzag

function zz = ZigZag8x8(quant)
ZigZag = [1 2 6 7 15 16 28 29;
          3 5 8 14 17 27 30 43;
          4 9 13 18 26 31 42 44;
          10 12 19 25 32 41 45 54;
          11 20 24 33 40 46 53 55;
          21 23 34 39 47 52 56 61;
          22 35 38 48 51 57 60 62;
          36 37 49 50 58 59 63 64];
      
zz1(ZigZag(:))=quant(:,:,1);
zz2(ZigZag(:))=quant(:,:,2);
zz3(ZigZag(:))=quant(:,:,3);

zz(:,1) = zz1;
zz(:,2) = zz2;
zz(:,3) = zz3;
end

function coeffs = DeZigZag8x8(zz)
ZigZag = [1 2 6 7 15 16 28 29;
          3 5 8 14 17 27 30 43;
          4 9 13 18 26 31 42 44;
          10 12 19 25 32 41 45 54;
          11 20 24 33 40 46 53 55;
          21 23 34 39 47 52 56 61;
          22 35 38 48 51 57 60 62;
          36 37 49 50 58 59 63 64];
      
zz1=zz(:,1);
zz2=zz(:,2);
zz3=zz(:,3);
      
quant(:,:,1)=zz1(ZigZag(:));
quant(:,:,2)=zz2(ZigZag(:));
quant(:,:,3)=zz3(ZigZag(:));

coeffs(:,:,1)=reshape(quant(:,:,1), 8, 8);
coeffs(:,:,2)=reshape(quant(:,:,2), 8, 8);
coeffs(:,:,3)=reshape(quant(:,:,3), 8, 8);
end


%% ZerorunEnc-dec

function zze = ZeroRunEnc_EoB(zz, EOB)
zz_length=length(zz);
blocknum=zz_length/64;
count=0;
zze=[];
i=1;
j=1;
while i<=blocknum*64
if zz(i)==0
    zze=[zze 0];
    x=i;
    if mod(x,64)==0
        j=j-1;
    end
    while x<=j*64
    if zz(x)==0
    count=count+1;
    x=x+1;
    else
        break
    end
    end
    if x==j*64+1
       zze(end)=EOB;
       j=j+1;
       i=(j-1)*64+1;
    else
        zze=[zze count-1];
        i=i+count;
    if mod(i,64)==0
        j=j+1;
    end
    end
else
    zze=[zze zz(i)];
    i=i+1;
    if mod(i,64)==0
        j=j+1;
    end
end
count=0;
end
end

function dst = ZeroRunDec_EoB(src, EoB)
% dst = [];
src_length=length(src);
block_num=1;
i=1;
j=1;
x=0;
while i<=src_length
    if src(i)~=EoB
        if src(i)==0
            dst(j)=0;
            if src(i+1)==0
                i=i+2;
                j=j+1;
            else
                x=src(i+1);
                dst(j+1:j+x)=0;
                i=i+2;
                j=j+x+1;
            end
        elseif 0<src(i)<block_num*64
            dst(j)=src(i);
            if mod(j,64)==0
                block_num=block_num+1;
            end
            i=i+1;
            j=j+1;
        end
    else
        dst(j:block_num*64)=0;
        i=i+1;
        j=block_num*64+1;
        block_num=block_num+1;
    end
end
end


%% Huffman

function [ BinaryTree, BinCode, Codelengths] = buildHuffman( p )
global y
p=p(:)/sum(p)+eps;              % normalize histogram
p1=p;                           % working copy
c=cell(length(p1),1);			% generate cell structure
for i=1:length(p1)				% initialize structure
    c{i}=i;
end

while size(c)-2					% build Huffman tree
    [p1,i]=sort(p1);			% Sort probabilities
    c=c(i);						% Reorder tree.
    c{2}={c{1},c{2}};           % merge branch 1 to 2
    c(1)=[];	                % omit 1
    p1(2)=p1(1)+p1(2);          % merge Probabilities 1 and 2
    p1(1)=[];	                % remove 1
end

%cell(length(p),1);              % generate cell structure
getcodes(c,[]);                  % recurse to find codes
code=char(y);

[numCodes, maxlength] = size(code); % get maximum codeword length

% generate byte coded huffman table
% code
length_b=0;
HuffCode=zeros(1,numCodes);
for symbol=1:numCodes
    for bit=1:maxlength
        length_b=bit;
        if(code(symbol,bit)==char(49)) HuffCode(symbol) = HuffCode(symbol)+2^(bit-1)*(double(code(symbol,bit))-48);
        elseif(code(symbol,bit)==char(48))
        else
            length_b=bit-1;
            break;
        end
    end
    Codelengths(symbol)=length_b;
end
BinaryTree = c;
BinCode = code;
clear global y;
return
    function getcodes(a,dum)
        if isa(a,'cell')                    % if there are more branches...go on
            getcodes(a{1},[dum 0]);    %
            getcodes(a{2},[dum 1]);
        else
            y{a}=char(48+dum);
        end
    end
end

function [bytestream] = enc_huffman_new( data, BinCode, Codelengths)
a = BinCode(data(:),:)';
b = a(:);
mat = zeros(ceil(length(b)/8)*8,1);
p  = 1;
for i = 1:length(b)
    if b(i)~=' '
        mat(p,1) = b(i)-48;
        p = p+1;
    end
end
p = p-1;
mat = mat(1:ceil(p/8)*8);
d = reshape(mat,8,ceil(p/8))';
multi = [1 2 4 8 16 32 64 128];
bytestream = sum(d.*repmat(multi,size(d,1),1),2);
end

function [output] = dec_huffman_new (bytestream, BinaryTree, nr_symbols)
output = zeros(1,nr_symbols);
ctemp = BinaryTree;
dec = zeros(size(bytestream,1),8);
for i = 8:-1:1
    dec(:,i) = rem(bytestream,2);
    bytestream = floor(bytestream/2);
end
dec = dec(:,end:-1:1)';
a = dec(:);

i = 1;
p = 1;
while(i <= nr_symbols)&&p<=max(size(a))
    while(isa(ctemp,'cell'))
        next = a(p)+1;
        p = p+1;
        ctemp = ctemp{next};
    end
    output(i) = ctemp;
    ctemp = BinaryTree;
    i=i+1;
end
end


%% SSD and SSD_rec

function motion_vectors_indices = SSD(ref_image, image)

wide = size(image(:,:,1),2);
high = size(image(:,:,1),1);
ref_image = padarray(ref_image,[4,4],0);% search boundary +-4

Index_Matrix = reshape(1:9^2,9,9)';
motion_vectors_indices = zeros(high/8,wide/8);
for i = 1:8:wide
    for j = 1:8:high
        current_block = image(j:j+7,i:i+7);
        bestSSE = 9999999;
        for x = i:i+8
            for y = j:j+8
                reference_block = ref_image(y:y+7,x:x+7);
                SSE = sum(sum((reference_block-current_block).^2));
                indexx = x-i+1;
                indexy = y-j+1;
                if SSE<bestSSE
                    bestSSE = SSE;
                    % always save best sse.
                    bestindexx = indexx;
                    bestindexy = indexy;
                end
            end
        end
        motion_vectors_indices((j-1)/8+1,(i-1)/8+1) = Index_Matrix(bestindexy,bestindexx);
    end
end

end

function rec_image = SSD_rec(ref_image, motion_vectors)

    wide = size(ref_image,2);
    high = size(ref_image,1);
    for i = 1:8:wide
        for j = 1:8:high
            y = ceil(motion_vectors((j-1)/8+1,(i-1)/8+1)/9)-5;
            if mod(motion_vectors((j-1)/8+1,(i-1)/8+1),9) == 0
                x = 4;
            else
                x = mod(motion_vectors((j-1)/8+1,(i-1)/8+1),9)-5;
            end
            if j+y>0 && i+x>0 && high>=j+y+7 && wide>=i+x+7
                rec_image(j:j+7,i:i+7,:) = ref_image(j+y:j+y+7,i+x:i+x+7,:);
            else
                rec_image(j:j+7,i:i+7,:) = ref_image(j:j+7,i:i+7,:);
                
            end
        end
    end
end

%% RGB and YCbCr

function yuv = ictRGB2YCbCr(rgb)
R = rgb(:,:,1);
G = rgb(:,:,2);
B = rgb(:,:,3);
Y = 0.299*R + 0.587*G + 0.114*B;
Cb = -0.169*R - 0.331*G + 0.5*B;
Cr = 0.5*R - 0.419*G - 0.081*B;
yuv(:,:,1) = Y;
yuv(:,:,2) = Cb;
yuv(:,:,3) = Cr;
end

function rgb = ictYCbCr2RGB(yuv)

Y = yuv(:,:,1);
Cb = yuv(:,:,2);
Cr = yuv(:,:,3);
R = Y + 1.402*Cr;
G = Y - 0.344*Cb - 0.714*Cr;
B = Y + 1.772*Cb;
rgb(:,:,1) = R;
rgb(:,:,2) = G;
rgb(:,:,3) = B;
end

%% Pmf 
function pmf = stats_marg(image, range)
pmf = hist(image(:),range);
pmf = pmf/sum(pmf);
end

%% PSNR and MSE
function PSNR = calcPSNR(Image, recImage)
PSNR = 10*log10((2^8-1)^2/calcMSE(Image, recImage));
end

function MSE = calcMSE(Image, recImage)

i1 = double(Image);% Input         : Image    (Original Image)
i2 = double(recImage);%   recImage (Reconstructed Image)
D = i1 - i2;
C = size(i1);
MSE = sum((D(:)).*(D(:)))/prod(C);% Output        : MSE      (Mean Squared Error)
end
