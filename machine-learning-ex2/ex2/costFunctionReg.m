function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for iter=1:m
    J = J - y(iter)*log(sigmoid(X(iter,:)*theta))-(1-y(iter))*log(1-sigmoid(X(iter,:)*theta));
end
J = J/m;
theta_sq = 0;
for iter=2:size(theta)
   theta_sq = theta_sq + power(theta(iter),2);
end
J = J + lambda*theta_sq/2/m;


for iter=1:m
   grad(1) = grad(1) + (sigmoid(X(iter,:)*theta) - y(iter))*X(iter,1);
end
for itr2=2:size(theta)
    for iter=1:m
       grad(itr2) = grad(itr2) + (sigmoid(X(iter,:)*theta) - y(iter))*X(iter,itr2);
    end
    grad(itr2) = grad(itr2) + lambda*theta(itr2);
end
grad = grad/m;

% =============================================================

end
