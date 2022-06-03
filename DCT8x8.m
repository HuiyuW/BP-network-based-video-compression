function coeff = DCT8x8(block)
    coeff(:,:,1) = dct2(block(:,:,1));
    coeff(:,:,2) = dct2(block(:,:,2));
    coeff(:,:,3) = dct2(block(:,:,3));
end