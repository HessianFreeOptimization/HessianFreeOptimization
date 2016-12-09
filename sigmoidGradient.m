function g = sigmoidGradient(z)
    g = sigmoid(z).*(1-sigmoid(z));
end