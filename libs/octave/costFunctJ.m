function J = costFunctJ (x,y,theta)

% x is the input training data set
% y is the result training data set
% slope, theta

m = size (x,1);

predictions = x*theta;

sqrErrors = (predictions-y).^2;

J = 1/(2*m) * sum(sqrErrors);
