function RMSECV_replace = get_replace_result(X,y,binary_matrix,A,fold)

binary_matrix_row = size(binary_matrix,1);
binary_matrix_col = size(binary_matrix,2);
RMSECV_replace = nan(binary_matrix_row, binary_matrix_col);
tic
for m = 1:binary_matrix_col 
    replace = binary_matrix;
    replace(binary_matrix(:,m)==1,m) = 0;
    replace(binary_matrix(:,m)==0,m) = 1;
    RMSECV_replace(:,m) = get_matrix_result( X,y,replace,A,fold);
    fprintf('The %d(th)/%d has finished, elapsed time is %g seconds!!\n', m,binary_matrix_col, toc)
end



















