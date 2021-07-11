function [X] = normalize(X)
%X:n*d
X(find(X<0))=0;
for i=1:size(X,1)
    if(norm(X(i,:))==0)
        
    else
        X(i,:) = X(i,:)./norm(X(i,:));
    end
end