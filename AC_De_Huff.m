function AC_decode = AC_De_Huff(AC_encode) %AC Huffman¸Ñ½X
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%«Ø¥ßAC Huffman¬ÛÃöªºtable
% L - category - total length - bsae code length -  base code
table = {...
 0  1   3  2 '00'
 0  2   4  2 '01'
 0  3   6  3 '100' 
 0  4   8  4 '1011'
 0  5  10  5 '11010' 
 0  6  12  6 '111000' 
 0  7  14  7 '1111000'  
 0  8  18 10 '1111110110' 
 0  9  25 16 '1111111110000010'
 0 10  26 16 '1111111110000011'
 1  1   5  4 '1100' 
 1  2   8  6 '111001' 
 1  3  10  7 '1111001' 
 1  4  13  9 '111110110' 
 1  5  16 11 '11111110110' 
 1  6  22 16 '1111111110000100'
 1  7  23 16 '1111111110000101'
 1  8  24 16 '1111111110000110'
 1  9  25 16 '1111111110000111'
 1 10  26 16 '1111111110001000'
 2  1   6  5 '11011' 
 2  2  10  8 '11111000'
 2  3  13 10 '1111110111' 
 2  4  20 16 '1111111110001001'
 2  5  21 16 '1111111110001010'
 2  6  22 16 '1111111110001011'
 2  7  23 16 '1111111110001100'
 2  8  24 16 '1111111110001101'
 2  9  25 16 '1111111110001110'
 2 10  26 16 '1111111110001111'
 3  1   7  6 '111010'
 3  2  11  9 '111110111'
 3  3  14 11 '11111110111' 
 3  4  20 16 '1111111110010000'
 3  5  21 16 '1111111110010001'
 3  6  22 16 '1111111110010010'
 3  7  23 16 '1111111110010011'
 3  8  24 16 '1111111110010100'
 3  9  25 16 '1111111110010101'
 3 10  26 16 '1111111110010110'
 4  1   7  6 '111011'
 4  2  12 10 '1111111000' 
 4  3  19 16 '1111111110010111'
 4  4  20 16 '1111111110011000'
 4  5  21 16 '1111111110011001'
 4  6  22 16 '1111111110011010'
 4  7  23 16 '1111111110011011'
 4  8  24 16 '1111111110011100'
 4  9  25 16 '1111111110011101'
 4 10  26 16 '1111111110011110'
 5  1   8  7 '1111010'
 5  2  12 10 '1111111001' 
 5  3  19 16 '1111111110011111'
 5  4  20 16 '1111111110100000'
 5  5  21 16 '1111111110100001'
 5  6  22 16 '1111111110100010'
 5  7  23 16 '1111111110100011'
 5  8  24 16 '1111111110100100'
 5  9  25 16 '1111111110100101'
 5 10  26 16 '1111111110100110'
 6  1   8  7 '1111011'
 6  2  13 11 '11111111000' 
 6  3  19 16 '1111111110100111'
 6  4  20 16 '1111111110101000'
 6  5  21 16 '1111111110101001'
 6  6  22 16 '1111111110101010'
 6  7  23 16 '1111111110101011'
 6  8  24 16 '1111111110101100'
 6  9  25 16 '1111111110101101'
 6 10  26 16 '1111111110101110'
 7  1   9  8 '11111001'
 7  2  13 11 '11111111001'
 7  3  19 16 '1111111110101111'
 7  4  20 16 '1111111110110000'
 7  5  21 16 '1111111110110001'
 7  6  22 16 '1111111110110010'
 7  7  23 16 '1111111110110011'
 7  8  24 16 '1111111110110100'
 7  9  25 16 '1111111110110101'
 7 10  26 16 '1111111110110110'
 8  1   9  8 '11111010'
 8  2  17 15 '111111111000000'
 8  3  19 16 '1111111110110111'
 8  4  20 16 '1111111110111000'
 8  5  21 16 '1111111110111001'
 8  6  22 16 '1111111110111010'
 8  7  23 16 '1111111110111011'
 8  8  24 16 '1111111110111100'
 8  9  25 16 '1111111110111101'
 8 10  26 16 '1111111110111110'
 9  1  10  9 '111111000' 
 9  2  18 16 '1111111110111111'
 9  3  19 16 '1111111111000000'
 9  4  20 16 '1111111111000001'
 9  5  21 16 '1111111111000010'
 9  6  22 16 '1111111111000011'
 9  7  23 16 '1111111111000100'
 9  8  24 16 '1111111111000101'
 9  9  25 16 '1111111111000110'
 9 10  26 16 '1111111111000111'
10  1  10  9 '111111001' 
10  2  18 16 '1111111111001000'
10  3  19 16 '1111111111001001'
10  4  20 16 '1111111111001010'
10  5  21 16 '1111111111001011'
10  6  22 16 '1111111111001100'
10  7  23 16 '1111111111001101'
10  8  24 16 '1111111111001110'
10  9  25 16 '1111111111001111'
10 10  26 16 '1111111111010000'
11  1  10  9 '111111010'
11  2  18 16 '1111111111010001'
11  3  19 16 '1111111111010010'
11  4  20 16 '1111111111010011'
11  5  21 16 '1111111111010100'
11  6  22 16 '1111111111010101'
11  7  23 16 '1111111111010110'
11  8  24 16 '1111111111010111'
11  9  25 16 '1111111111011000'
11 10  26 16 '1111111111011001'
12  1  11 10 '1111111010' 
12  2  18 16 '1111111111011010'
12  3  19 16 '1111111111011011'
12  4  20 16 '1111111111011100'
12  5  21 16 '1111111111011101'
12  6  22 16 '1111111111011110'
12  7  23 16 '1111111111011111'
12  8  24 16 '1111111111100000'
12  9  25 16 '1111111111100001'
12 10  26 16 '1111111111100010'
13  1  12 11 '11111111010' 
13  2  18 16 '1111111111100011'
13  3  19 16 '1111111111100100'
13  4  20 16 '1111111111100101'
13  5  21 16 '1111111111100110'
13  6  22 16 '1111111111100111'
13  7  23 16 '1111111111101000'
13  8  24 16 '1111111111101001'
13  9  25 16 '1111111111101010'
13 10  26 16 '1111111111101011'
14  1  13 12 '111111110110'
14  2  18 16 '1111111111101100'
14  3  19 16 '1111111111101101'
14  4  20 16 '1111111111101110'
14  5  21 16 '1111111111101111'
14  6  22 16 '1111111111110000'
14  7  23 16 '1111111111110001'
14  8  24 16 '1111111111110010'
14  9  25 16 '1111111111110011'
14 10  26 16 '1111111111110100'
15  1  17 16 '1111111111110101'
15  2  18 16 '1111111111110110'
15  3  19 16 '1111111111110111'
15  4  20 16 '1111111111111000'
15  5  21 16 '1111111111111001'
15  6  22 16 '1111111111111010'
15  7  23 16 '1111111111111011'
15  8  24 16 '1111111111111100'
15  9  25 16 '1111111111111101'
15 10  26 16 '1111111111111110'
 0  0   4  4 '1010'    %EOB
15  0  12 12 '11111111011'};  %ZRL

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i =1;  %¬ö¿ý¥Ø«e²¾¨ì¤@³s¦ê½s½Xªº¦ó³B(½s½X´å¼Ð)
count=1; %decode°}¦C¤¤ªº²¾°Ê´å¼Ð(¸Ñ½X´å¼Ð)
AC_decode = [];
while i < length(AC_encode) %­Y´å¼Ð¤ñ½s½Xªø«×ÁÙµu¡A«h«ùÄò°µ¸Ñ½X
    if i >= length(AC_encode)-25  %·í´å¼Ð²¾¨ì±µªñ½s½Xªø«×§ÀºÝªº¦a¤è
        temp = AC_encode(i:length(AC_encode)); 
    else
        temp = AC_encode(i:25+i);  %¨ú26 bits¥X¨Ó¸Ñ½X
    end  
    if strcmp(temp(1:4) ,'1010')  %§PÂ_¬O§_¬°EOB
        a = 63 - mod(length(AC_decode),63); %§PÂ_»Ý­n¶ñ¤J´X­Ó0
        if  a>0   %­Y»Ý­n¶ñ¤J0
          AC_decode(count:count+a-1) = 0;
          count = count+a; %¸Ñ½X´å¼Ð+a
        end
        i = i+4; %½s½X´å¼Ð²¾°Ê4
    elseif length(temp)>=12 && strcmp(temp(1:12) , '111111110111') %§PÂ_¬O§_¬°ZRL
           AC_decode(count:count+15) = 0; %¶ñ¤J16­Ó0
           count = count+16; %¸Ñ½X´å¼Ð+16
           i = i+12; %½s½X´å¼Ð²¾°Ê12
    else 
       for k = 1:160 %±qtable 160¦C¤¤¥h§ä´M¹ïÀ³ªºcode
           if length(temp) >= table{k,4}
             if strcmp(table{k,5} , temp(1:table{k,4})) %¥ý§PÂ_«e­±ªºbase code¬O§_¬Û¦P
               if table{k,1}>0 %­Ybase code¬Û¦P¡A¥ý¶ñ¤JL­Ó0
                   AC_decode(count:count+table{k,1}-1) = 0;
                   count=count+table{k,1};
               end
               
               %¦A­pºâªþ¥[¦ì¤¸©Ò¹ïÀ³ªºnumber
               AC_decode(count) = -2^(table{k,2}) +1 + bin2dec(temp(table{k,4}+1:table{k,3}));
               if bin2dec(temp(table{k,4}+1:table{k,3})) >= 2^(table{k,2}-1)  %§PÂ_¬O§_ÄÝ©ó¥¿ªº¥k¥bÃä
                   AC_decode(count) = AC_decode(count)+2^(table{k,2})-1;
               end          

               count = count+1; %¸Ñ½X´å¼Ð+1
               i = i+table{k,3}; %½s½X´å¼Ð ²¾°Ê½s½Xªºtotal length
               break; %°µ§¹§Y¥i¸õ¥X°j°é  
             end
           end
       end
    end
end  % end of while
end
