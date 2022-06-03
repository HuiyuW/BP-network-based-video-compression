% Combination_the_Matrix
function [YY CbCb CrCr] = MatCom_fn(DC_decode,AC_decode,width,height,qScale)
%% Parameters
seq=[9,2,3,10,17,25,18,11,4,5,12,19,26,33,41,34,27,20,13,6,7,14,21,28,35,42,49,57,50,43,36,29,22,15,8,16,23,30,37,44,51,58,59,52,45,38,31,24,32,39,46,53,60,61,54,47,40,48,55,62,63,56,64];
% ¡ô zigzag sequence

%% 
% L1 = length(DC_decode)/1.5;  % 4096  #of DC elements for YQ1
% L2 = length(AC_decode)/1.5;  % 258048 #of AC elements for YQ1
% sizeY = sqrt(L1+L2);  %512

A = zeros(8);
k=1;  
for i=1:width/8
    XX = [8*i-7:8*i];
    for j =1:height/8
A(1,1) = DC_decode(k);
A(seq) = AC_decode(63*k-62:63*k);
% DeQuant8x8()
dequant = DeQuantL8x8_q(A,qScale); 
% Idct 
I_dct =idct2(dequant);
YY(XX,8*j-7:8*j) = I_dct;
% YY(XX,8*j-7:8*j) = Cf' * (A.*Qy) * Cf;  %% De_DCT_2n
k = k+1;
    end
end

% L3 = length(DC_decode)/6;  % 1024
% L4 = length(AC_decode)/6;  % 64512
% sizeY = sqrt(L3+L4); %256

B = zeros(8);
C = zeros(8);
li = (width/8)*(height/8)*2+1;
for i=1:width/8
    XX = [8*i-7:8*i];
    for j =1:height/8
B(1,1) = DC_decode(k);
B(seq) = AC_decode(63*k-62:63*k);

C(1,1) = DC_decode(li);
C(seq) = AC_decode(63*li-62:63*li);

% DeQuant8x8()
dequantB = DeQuantC8x8_q(B,qScale); 
% Idct 
I_dctB =idct2(dequantB);
CbCb(XX,8*j-7:8*j) = I_dctB;
k = k+1;
% DeQuant8x8()
dequantC = DeQuantC8x8_q(C,qScale); 
% Idct 
I_dctC =idct2(dequantC);
CrCr(XX,8*j-7:8*j) = I_dctC;


% CbCb(XX,8*j-7:8*j) = Cf' * (B.*Qc) * Cf;
% k = k+1;
% CrCr(XX,8*j-7:8*j) = Cf' * (C.*Qc) * Cf;
li = li+1;
    end
end
end

