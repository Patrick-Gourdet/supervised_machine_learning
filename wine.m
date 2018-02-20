%LLS classifier for the simplified iris dataset

B = fopen('wine.txt','r');

%%%
% This computes the number of rows
 c = 0;
fgetl(B);
while ~feof(B)
fgetl(B);
c = c+1;
end
numrows = c;

%%%
B = fopen('wine.txt','r');
C = textscan(B,'%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f ,%f');

col5 = C{1}; %tags
col1 = C{2};
col2 = C{3};
col3 = C{4};
col4 = C{5}; 
col8 = C{6};
col9 = C{7};
col10 = C{8};
col11 = C{9};
col12 = C{10};
col13 = C{11};
col14 = C{12};
col15 = C{13};
col16 = C{14};

training = 8;  %  specify first n rows to be used as training.  
%For training purposes I mixed up the original file a bit so that all the
%different species of iris would be represented in the training set. 

col6=zeros(3,1,training);

for n=1:numrows  % this loop assigns numerical tags
    if col5(n)==1
        col6(1,1,n)=1;
    elseif col5(n)==2
        col6(2,1,n)=1;
    else
        col6(3,1,n)=1;     
    end
end

constraint = zeros(numrows+1,1);
constraint = constraint + 1;

x = [constraint col1 col2 col3 col4 col8 col9 col10 col11 col12 col13 col14 col15 col16];  %data row vector

lhs = zeros(14,14,training); %these are some initiializations
lhstemp = 0;
rhs = zeros(14,3,training);
rhstemp = [1:14];

for i=1:training
    lhs(:,:,i)=x(i,:)'*x(i,:);      %compute lhs of w-hat
    lhstemp = lhs(:,:,i)+ lhstemp;  %summing lhs
end
lhstemp= inv(lhstemp);   %taking inverse

for i=1:training
    rhs(:,:,i)=x(i,:)'*col6(:,:,i)';   %compute rhs of w-hat
end

rhsfinal=zeros(14,3);

for i=1:training  %summation of RHS
    rhsfinal = rhsfinal + rhs(:,:,i);
end

w=lhstemp*rhsfinal;  %  this is w-hat from the project notes
w2=w';  

prediction = zeros(3,1,numrows);
results = [1:numrows];
    

for i=1:numrows    %  w'*x
    prediction(:,:,i) = w2*x(i,:)';
    current=prediction(:,:,i);
    [value,results(i)] = max(current(:));
end

col7 = zeros(3,1,numrows);

for i=1:numrows  %  Assigns the predicted y_i
    if results(i)==1
        col7(:,:,i)=[1,0,0]';
    elseif results(i)==2
        col7(:,:,i)=[0,1,0]';
    else 
        col7(:,:,i)=[0,0,1]';
    end
end

count = 0;
comp = [training:numrows];

for i=training+1:numrows    %compare the predicted tag to the actual label
    temp = col6(:,:,i)-col7(:,:,i);
    comp(i)=norm(temp);
    if norm(temp)==0
        count = count+1;
    end
end

percentage = count/(numrows-training) %percentage in the test set correct