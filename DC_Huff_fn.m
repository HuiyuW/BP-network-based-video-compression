% DC_Huff_fn
function DC_Huff = DC_Huff_fn(DC_dif)
% Log-scale to classify
%Category     	DC_diff Region	       Huffman Code	   Code Length
%  0	            0	                   00	          2
%  1	          -1,1	                  010	          3
%  2	       -3, -2, 2, 3	              011	          3
%  3	       -7~-4, 4~7	              100	          3
%  4	      -15~-8, 8~15	              101	          3
%  5	     -31~-16, 16~31	              110	          3
%  6	     -63~-32, 32~63	             1110	          4
%  7	    -127~-64, 64~127	        11110	          5
%  8	    -255~-128, 128~255	       111110	          6
%  9	    -511~-256, 256~511	      1111110	          7
% 10	   -1023~-512, 512~1023	     11111110	          8
% 11	   -2047~-1023, 1023~2047	111111110	          9
L = length(DC_dif);
Huff_Table = {'00','010','011','100','101','110','1110','11110','111110','1111110',...
             '11111110','111111110'};
DC_Huff=[];
for i=1:L
if DC_dif(i)==0
    mycode = '00';
elseif DC_dif(i)>0
    index = floor(log2(DC_dif(i)));
    mycode = strcat(Huff_Table{index+2},dec2bin(DC_dif(i)));
    
elseif DC_dif(i)<0
    index = floor(log2(abs(DC_dif(i))));
    bdc = dec2bin(abs(DC_dif(i)));
    for j=1:length(bdc)
        if(bdc(j)=='0') bdc(j)='1'; %Complement
        else bdc(j)='0';
        end
    end
    mycode = strcat(Huff_Table{index+2},bdc);
end
DC_Huff = strcat(DC_Huff,mycode);
end




    