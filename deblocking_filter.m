function filtered_image = deblocking_filter(image,index)
%% Output
Alphatable = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,5,6,7,8,9,10,12,13,15,17,20,22,...
    25,28,32,36,40,45,50,56,63,71,80,90,101,113,127,144,162,182,203,226,255,255];
Batatable = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,3,3,3,3,4,4,4,6,6,7,7,8,8,9,9,...
    10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18];
alpha = Alphatable(index);
beta = Batatable(index);
sz = size(image);
filtered_image = image;
blocksize = 8;
%% vertical direction 
for channel = 1:sz(3)
    for row = blocksize: blocksize:sz(1)-blocksize
        for column = 1:sz(2)
            %% p0-3 & q 0-3
            %% vertical direction: column fixed 
            p0 = image(row,column,channel);
            p1 = image(row-1,column,channel);
            p2 = image(row-2,column,channel);
            p3 = image(row-3,column,channel);
            q0 = image(row+1,column,channel);
            q1 = image(row+2,column,channel);
            q2 = image(row+3,column,channel);
            q3 = image(row+4,column,channel);
            diff1 = abs(p0 - q0);
            diff2 = abs(p2 - p0);
            diff3 = abs(q2 - q0);
            diff4 = abs(p1 - p0);
            diff5 = abs(q1 - q0);
            if (diff1<alpha) && (diff4<beta) && (diff5<beta) && ...
                    (diff2<beta) && (diff3<beta) % p,q strong deblocking
                p0_new = (p2+2*p1+2*p0+2*q0+q1+4)/8;
                p1_new = (p2+p1+p0+q0+2)/4;
                p2_new = (2*p3+3*p2+p1+p0+q0+4)/8;
                q0_new = (q2+2*q1+2*q0+2*p0+p1+4)/8;
                q1_new = (q2+q1+q0+p0+2)/4;
                q2_new = (2*q3+3*q2+q1+q0+p0+4)/8;
                filtered_image(row-2:row+3,column,channel) =[p2_new,p1_new,p0_new,...
                    q0_new,q1_new,q2_new];
            elseif (diff1<alpha) && (diff4<beta) && (diff5<beta) && ...
                    (diff2>=beta) && (diff3<beta) %q strong p weak
                p0_new = (2*p1+p0+q1+2)/4;
                q0_new = (q2+2*q1+2*q0+2*p0+p1+4)/8;
                q1_new = (q2+q1+q0+p0+2)/4;
                q2_new = (2*q3+3*q2+q1+q0+p0+4)/8;
                filtered_image(row:row+3,column,channel) =[p0_new,q0_new,q1_new,q2_new];
                
            elseif (diff1<alpha) && (diff4<beta) && (diff5<beta) && ...
                    (diff2<beta) && (diff3>= beta) %p strong q weak
                p0_new = (p2+2*p1+2*p0+2*q0+q1+4)/8;
                p1_new = (p2+p1+p0+q0+2)/4;
                p2_new = (2*p3+3*p2+p1+p0+q0+4)/8;
                q0_new = (2*q1+q0+p1+2)/4;
                filtered_image(row-2:row+1,column,channel) =[p2_new,p1_new,p0_new,q0_new];
            elseif (diff1<alpha) && (diff4<beta) && (diff5<beta)% weak deblocking
                p0_new = (2*p1+p0+q1+2)/4;
                q0_new = (2*q1+q0+p1+2)/4;
                filtered_image(row:row+1,column,channel) = [p0_new,q0_new];
            end   
        end
    end
end