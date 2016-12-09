function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    % Setup some useful variables
    m = size(X, 1);

    % You need to return the following variables correctly 
    J = 0;

    % feedforward & calculate cost: J
    X = [ones(m,1), X];
    z2 = X*Theta1';
    a2 = [ones(m,1), sigmoid(z2)];
    h = sigmoid(a2*Theta2');

    for k = 1:num_labels
        c = (y == k);
        J = J - (c'*log(h(:,k))+(1-c)'*log(1-h(:,k)));
    end

    J = J/m;

    add = sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end)));
    add = add*lambda/m/2;
    J = J + add;

    % calculate grad
    De1 = zeros(size(Theta1));
    De2 = zeros(size(Theta2));
    for t = 1:m
        delta3 = h(t,:)' - ((1:num_labels)' == y(t));
        delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2(t,:)');
        De1 = De1 + delta2*X(t,:);
        De2 = De2 + delta3*a2(t,:);
    end

    Theta1_grad = De1/m;
    Theta2_grad = De2/m;

    Theta1_grad = Theta1_grad + lambda/m*[zeros(hidden_layer_size,1),Theta1(:,2:end)];
    Theta2_grad = Theta2_grad + lambda/m*[zeros(num_labels,1), Theta2(:,2:end)];

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
