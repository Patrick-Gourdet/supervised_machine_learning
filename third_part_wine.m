A = csvread('wine.csv');

[m,n]=size(A);

y = A(:,1); 
x = A(:,2:n);
num = ceil(0.1*m);

perm = randperm(length(x));
x = x(perm,:);
y = y(perm,:);

xt = x(1:num,: );
yt = y(1:num,:);

t = templateSVM('Standardize',1);
%Standard onevone
Mdl = fitcecoc(x,y,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'},'Verbose',2);
% One v All coding
Mdl2 = fitcecoc(x,y,'Coding','onevsall','Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'},'Verbose',2);
CVMdl = crossval(Mdl);
oosLoss = kfoldLoss(CVMdl)

[label,~,~,Posterior] = resubPredict(Mdl,'Verbose',1);
[label2,~,~,Posterior2] = resubPredict(Mdl2,'Verbose',1);
Mdl.BinaryLoss;
idx = randsample(size(x,1),10,1);
Mdl.ClassNames
table(y(idx),label(idx),Posterior(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})
Mdl2.ClassNames
table(y(idx),label2(idx),Posterior2(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})
