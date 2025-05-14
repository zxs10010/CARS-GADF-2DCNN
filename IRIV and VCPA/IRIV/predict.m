function [RMSEP,RMSEF]=predict(Xtrain,ytrain,Xtest,ytest,selected_variables,A,fold,method)

%++ prediction with the selected variables

if nargin<8;method='center';end;
if nargin<7;fold=10;end;
if nargin<6;A=10;end;


Xtrain=Xtrain(:,selected_variables);
Xtest=Xtest(:,selected_variables);
CV=plscvfold(Xtrain,ytrain,A,fold,method);
A_opt=CV.optPC; 
PLS=pls(Xtrain,ytrain,A_opt,method);
Xtest_expand=[Xtest ones(size(Xtest,1),1)];
coef=PLS.coef_origin;
ypred=Xtest_expand*coef(:,end);
%%%%%%%%%   B E CAUTIOUS  ####################   
RMSEF=sqrt(PLS.SSE/size(Xtrain,1));
RMSEP=sqrt(sum((ytest-ypred).^2)/size(Xtest,1));
