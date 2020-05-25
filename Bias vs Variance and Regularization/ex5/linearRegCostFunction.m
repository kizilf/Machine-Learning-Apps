function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hx = X * theta;

J = (hx - y).^2;
J = sum(J);

J = J / (2*m);

coef = lambda / (2*m);

J = J + (coef * sum(theta(2:end).^2));


for i=1:size(theta,1)
    grad(i) = sum((hx-y).*X(:,i));
    grad(i) = grad(i) / m;
    if(i ~= 1)
        grad(i) = grad(i) + (lambda/m) * theta(i);
    end;
end;
    

% =========================================================================

grad = grad(:);

end
