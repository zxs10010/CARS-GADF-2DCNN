%+++ Example

%load('diseul.mat')
fnames='shui';
load(fnames)


for i=26:35
    F=IRIV(Xtrain,ytrain,10,5,'center');

    [RMSEP,RMSEF]=predict(Xtrain,ytrain,Xtest,ytest,F.SelectedVariables,10,5,'center')

    file_name = ['E:\feature selection\IRIV_1.1.1\IRIV_1.1\',fnames,'_select_',num2str(i)];
    save(file_name)
end
