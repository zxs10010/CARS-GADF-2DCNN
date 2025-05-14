function binary_matrix = generate_binary_matrix(X_num, row)
%+++ Input:  X_num: the number of vairbales in the X matrix
%            row: the number of the row of binary matrix
%           
%+++ Output: binary_matrix: the matrix contains 0 and 1.

rand_binary = nan(row, X_num);
control = 0;
while control == 0

    for n = 1:X_num
        rand_row = randperm(row);
        rand_row(rand_row<=length(rand_row)/2) = 0;
        rand_row(rand_row>0) = 1;
        rand_binary(:,n) = rand_row';
    end

    for m = 1:row
        if sum(rand_binary(m,:)) <= 1
            control = 0;
            break; 
        else
            control = 1;
        end
    end

end
rand_binary(rand_binary==0) = 0;
binary_matrix=rand_binary;

