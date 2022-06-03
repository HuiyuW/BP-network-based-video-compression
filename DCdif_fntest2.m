% DC differential encoding (Interlaced Scanning   Y => Cb => Cr )
function [DC  AC] = DCdif_fn(Y,Cb,Cr,m,n)
seq=[9 2 3 10 17 25 18 11 4 5 12 19 26 33 ...
     41 34 27 20 13 6 7 14 21 28 35 42 49 57 50 ...
     43 36 29 22 15 8 16 23 30 37 44 51 58 59 52 ...
     45 38 31 24 32 39 46 53 60 61 54 47 40 48 55 ...
     62 63 56 64];
k=1; AC = cell(1,9000);

li=1;
for i=1:(m/8)    %first columns then rows
    XX = 8*i-7;
    KK = [8*i-7:8*i];
    for j=1:(n/8)
    DC(li) = Y(XX,8*j-7); li=li+1;
    B = Y(KK,8*j-7:8*j);    
    AC(k) = { B(seq)} ; k=k+1;
    end
end


for i=1:(m/8)    %first columns then rows
    XX = 8*i-7;
    KK = [8*i-7:8*i];
    for j=1:(n/8)
    DC(li) = Cb(XX,8*j-7); li=li+1;
    B = Cb(KK,8*j-7:8*j);    
    AC(k) = { B(seq)}; k=k+1;
    end
end
for i=1:(m/8)    %first columns then rows
    XX = 8*i-7; KK = [8*i-7:8*i];
    for j=1:(n/8)
    DC(li) = Cr(XX,8*j-7); li=li+1;
    B = Cr(8*i-7:8*i,8*j-7:8*j);    
    AC(k) = { B(seq)}; k=k+1;
    end
end
AC = cell2mat(AC);


