function  X=Ni(A)
%Input  - A is an N x N matrix
%Output - I is an N x N inverse matrix of A 
%and I(j,:)containing the solution to AX(:,j) =E(:,j).
%Initialize X, Y,the temporary storage matrix C, and the row 
% permutation information matrix R
A = gpuArray(A);
[N,N]=size(A);
B=gpuArray(eye(N));   %B is an N x N identity matrix
X=gpuArray(zeros(N,N));
Y=gpuArray(zeros(N,N));
C=gpuArray(zeros(1,N));
R=1:N;
%the next steps is to find the factorization(factorize for only once)
for p=1:N-1
%Find the pivot row for column p
   [max1, j]=max(abs(A(p:N,p)));
%Interchange row p and j
      C=A(p,:);
      A(p,:)=A(j+p-1,:);
      A(j+p-1,:)=C;
      d=R(p);
      R(p)=R(j+p-1);
      R(j+p-1)=d;
      if A(p,p)==0
      'A is singular.  No unique solution'
      break
      end
   %Calculate multiplier and place in subdiagonal portion of A
      for k=p+1:N
         mult=A(k,p)/A(p,p);
     A(k,p) = mult;
         A(k,p+1:N)=A(k,p+1:N)-mult*A(p,p+1:N);
      end
end
for j=1:N    
    %when j is fixed then the method is similar to the Program 3.3
    %Solve for Y(:,j)
    Y(1,j) = B(R(1),j);
    for k=2:N
        Y(k,j)= B(R(k),j)-A(k,1:k-1)*Y(1:k-1,j);
    end
    %Solve for X(:,j)
    X(N,j)=Y(N,j)/A(N,N);
    for k=N-1:-1:1
        X(k,j)=(Y(k,j)-A(k,k+1:N)*X(k+1:N,j))/A(k,k);
    end
end
X = gather(X);