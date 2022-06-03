% DC_De_Huff
function DC_decode_diff = DC_De_Huff(DC_Huff) %AC Huffman¸Ñ½X
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
L = length(DC_Huff);        DC_decode_diff = [];
% Huff_Table = {'00','010','011','100','101','110','1110','11110','111110','1111110','11111110','111111110'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i =1;  %¬ö¿ý¥Ø«e²¾¨ì¤@³s¦ê½s½Xªº¦ó³B(½s½X´å¼Ð)
while i < L %­Y´å¼Ð¤ñ½s½Xªø«×ÁÙµu¡A«h«ùÄò°µ¸Ñ½X
    count = 1; 
    if i+count<=L && strcmp(DC_Huff(i:i+count), '00')==1   DC_decode_diff = [DC_decode_diff, 0]; i=i+2;
    end
    count = 2;
    if  i+count<=L && strcmp(DC_Huff(i:i+count),'010')==1
        if DC_Huff(i+count+1)=='1'  DC_decode_diff = [DC_decode_diff ,1]; i=i+4;
        else  DC_decode_diff = [DC_decode_diff, -1]; i=i+4;
        end
    elseif i+count<=L && strcmp(DC_Huff(i:i+count),'011')==1
        if DC_Huff(i+count+1)=='1' DC_decode_diff = [DC_decode_diff ,bin2dec(DC_Huff(i+count+1:i+count+2))]; i=i+5;
        else DC_decode_diff = [DC_decode_diff , bin2dec(DC_Huff(i+count+1:i+count+2))-3]; i=i+5;
        end
    elseif i+count<=L && strcmp(DC_Huff(i:i+count),'100')==1
        if DC_Huff(i+count+1)=='1' DC_decode_diff = [DC_decode_diff ,bin2dec(DC_Huff(i+count+1:i+count+3))]; i=i+6;
        else DC_decode_diff = [DC_decode_diff , bin2dec(DC_Huff(i+count+1:i+count+3))-7]; i=i+6;
        end
    elseif i+count<=L && strcmp(DC_Huff(i:i+count),'101')==1
        if DC_Huff(i+count+1)=='1'  DC_decode_diff = [DC_decode_diff ,bin2dec(DC_Huff(i+count+1:i+count+4))]; i=i+7;
        else DC_decode_diff = [DC_decode_diff  ,bin2dec(DC_Huff(i+count+1:i+count+4))-15]; i=i+7;
        end
    elseif i+count<=L && strcmp(DC_Huff(i:i+count),'110')==1
        if DC_Huff(i+count+1)=='1' DC_decode_diff = [DC_decode_diff ,bin2dec(DC_Huff(i+count+1:i+count+5))]; i=i+8;
        else DC_decode_diff = [DC_decode_diff , bin2dec(DC_Huff(i+count+1:i+count+5))-31]; i=i+8;
        end
    end
    count = 3;
    if i+count<=L && strcmp(DC_Huff(i:i+count),'1110')==1
        if DC_Huff(i+count+1)=='1' DC_decode_diff = [DC_decode_diff ,bin2dec(DC_Huff(i+count+1:i+count+6))]; i=i+10;
        else DC_decode_diff = [DC_decode_diff  ,bin2dec(DC_Huff(i+count+1:i+count+6))-63]; i=i+10;
        end
    end
    count = 4;
    if i+count<=L && strcmp(DC_Huff(i:i+count),'11110')==1
        if DC_Huff(i+count+1)=='1' DC_decode_diff = [DC_decode_diff ,bin2dec(DC_Huff(i+count+1:i+count+7))]; i=i+12;
        else DC_decode_diff = [DC_decode_diff  ,bin2dec(DC_Huff(i+count+1:i+count+7))-127]; i=i+12;
        end
    end
    count = 5; 
    if i+count<=L && strcmp(DC_Huff(i:i+count),'111110')==1
        if DC_Huff(i+count+1)=='1' DC_decode_diff = [DC_decode_diff, bin2dec(DC_Huff(i+count+1:i+count+8))]; i=i+14;
        else DC_decode_diff = [DC_decode_diff , bin2dec(DC_Huff(i+count+1:i+count+8))-225]; i=i+14;
        end
    end
    count = 6; 
    if i+count<=L && strcmp(DC_Huff(i:i+count),'1111110')==1
        if DC_Huff(i+count+1)=='1' DC_decode_diff = [DC_decode_diff ,bin2dec(DC_Huff(i+count+1:i+count+9))]; i=i+16;
        else DC_decode_diff = [DC_decode_diff  ,bin2dec(DC_Huff(i+count+1:i+count+9))-511]; i=i+16;
        end
    end
    count = 7; 
    if i+count<=L && strcmp(DC_Huff(i:i+count),'11111110')==1 
        if DC_Huff(i+count+1)=='1' DC_decode_diff = [DC_decode_diff ,bin2dec(DC_Huff(i+count+1:i+count+10))]; i=i+18;
        else DC_decode_diff = [DC_decode_diff  ,bin2dec(DC_Huff(i+count+1:i+count+10))-1023]; i=i+18;
        end
    end
    count = 8; 
    if i+count<=L && strcmp(DC_Huff(i:i+count),'111111110')==1 && i+count<L
        if DC_Huff(i+count+1)=='1'  DC_decode_diff = [DC_decode_diff ,bin2dec(DC_Huff(i+count+1:i+count+11))]; i=i+20;
        else DC_decode_diff = [DC_decode_diff  ,bin2dec(DC_Huff(i+count+1:i+count+11))-2047]; i=i+20;    
        end  
    end
end
end

