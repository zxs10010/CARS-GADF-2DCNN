function F=IRIV(X,y,A_max,fold,method)
%     IRIV: Iteratively Retaining Informative Variables
%+++ Input:  X: m x n  (Sample matrix)
%            y: m x 1  (measured property)
%            A_max: The max PC for cross-validation
%            fold: the group number for cross validation. when fold = m, it is leave-one-out CV
%            method: pretreatment method. Contains: autoscaling, center etc.

%+++ Yonghuan Yun, June. 2, 2013, yunyonghuan@foxmail.com
%+++ Advisor: Yizeng Liang, yizeng_liang@263.net

%++++Ref:  Yong-Huan Yun, Wei-Ting Wang, Hong-Dong Li, Yizeng Liang, Qingsong Xu, A strategy that iteratively 
%          retains informative variables for selecting optimal variable subset in multivariate calibration, 
%           Anal Chim Acta, 2014. http://dx.doi.org/10.1016/j.aca.2013.11.032

%+++       Hongyan Zhang, Haiyan Wnag. Improving  accuracy for cancer classificationwith a new algorithm for 
%          gene selection.  BMC Bioinformatics, November 2012, 13:298


CV=plscvfold(X,y,A_max,fold,method);
A=CV.optPC;               % Choose the optimal principle components for PLS 
XX=X;
[~,Nx]=size(X);   
time=0; 
tic;
varnumber=1:Nx;
j=1;
remain_number(j)=Nx;
control=0;
while control==0      
    [~,Nx]=size(X);
    % determine the row number of binary matrix  based on the number of retained variables of each round      
    if Nx>=500; row=500; end      
    if Nx>=300&&Nx<500; row=300; end      
    if Nx>=100&&Nx<300; row=200; end       
    if Nx>=50&&Nx<100; row=100; end       
    if Nx>=10&&Nx<50; row=50; end    
    
    RMSECV=zeros(row,1);      
    RMSECV_origin=zeros(row,Nx);      
    RMSECV_replace=zeros(row,Nx);          
    binary_matrix=generate_binary_matrix(Nx, row);      
    RMSECV = get_matrix_result( X,y,binary_matrix,A,fold);     
    RMSECV_origin=repmat(RMSECV,1,Nx);     
    RMSECV_replace = get_replace_result(X,y,binary_matrix,A,fold);                    
    RMSECV_exclude = RMSECV_replace;      
    RMSECV_include = RMSECV_replace;      
    RMSECV_exclude(binary_matrix==0) = RMSECV_origin(binary_matrix==0);       
    RMSECV_include(binary_matrix==1) = RMSECV_origin(binary_matrix==1);          
    DMEAN=zeros(1,Nx);       
    Pvalue=zeros(1,Nx);       
    H=zeros(1,Nx);
    for i=1:Nx()           
        %+++ Mann-Whitney U test for variable assessment           
        [p,h] = ranksum(RMSECV_exclude(:,i),RMSECV_include(:,i),'alpha',0.05);           
        Pvalue(i)=p;H(i)=h;          
        temp_DMEAN=mean(RMSECV_exclude(:,i))-mean(RMSECV_include(:,i));        
        %+++ Just a trick, indicating uninformative and interfering variable if Pvalue>1.         
        if temp_DMEAN<0;Pvalue(i)=p+1;end;                
        DMEAN(i)=temp_DMEAN;      
    end   
    ST{j}=H;      
    strong{j}=intersect(varnumber(H==1),varnumber(Pvalue<1));        
    weak{j}=intersect(varnumber(H==0),varnumber(Pvalue<1));     
    uinformative{j}=intersect(varnumber(H==0),varnumber(Pvalue>1));      
    interfering{j}=intersect(varnumber(H==1),varnumber(Pvalue>1));           
    time=time+toc;      
    remove_variables{j}=varnumber(Pvalue>1);
    store=zeros(2,length(varnumber));
    store(1,:)=varnumber;
    store(2,:)=Pvalue;
    P{j}=store;
    j=j+1; 
    remain_number(j)=sum(Pvalue<1);   
    
    % observe whether there are uninformative and intefering variables or not               
    if sum(Pvalue>=1)>0              
        varnumber(Pvalue>=1)=[]; 
        X=XX(:,varnumber);
        fprintf('The %d th round of IRIV has finished!  ', j-1)
        fprintf('Remain %d / %d  variable, using time: %g seconds!\n', sum((Pvalue<1) ),Nx, time)
    else control=1;       
    end  
end
fprintf('The iterative rounds of IRIV have been finished, now enter into the process of backward elimination!\n')

  % backward elimination after several iterative rounds   
variables = backward_elimination(X,y,A,fold,method,varnumber);     
F.SelectedVariables=variables;
F.Remain_number=remain_number; % the number of remained variables in each round
F.time=time;
F.Pvalue=P;


