function RMSECV = get_matrix_result(X,y,binary_matrix,A,fold)


binary_matrix_row = size(binary_matrix,1);
for m = 1:binary_matrix_row
    temp = binary_matrix(m,:);
    del_X = temp==0;
    X_new = X;
    X_new(:,del_X) = [];
    CV=plscvfold(X_new,y,A,fold,'center');
    RMSECV(m)=CV.RMSECV;
%  fprintf('The %d(th) circle 1 has finished, elapsed time is %g seconds!!\n', m, toc)
end
RMSECV=RMSECV';
   