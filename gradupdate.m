function [xn,vn,mt,vt,diagG,Eg2,Et2] = gradupdate(strategy,grad,xo,xoo,vo,diagG,Eg2,Et2,...
    mt, vt, i, eta,gamma,beta1,beta2,epsilon)
    vn = gamma*vo + eta*grad;
    diagG = diagG + grad.*grad;
    Eg2 = gamma*Eg2 + (1-gamma)*(grad.*grad);
    Et2 = gamma*Et2 + (1-gamma)*((xo-xoo).*(xo-xoo));
    mt = beta1*mt + (1-beta1)*grad;
    vt = beta2*vt + (1-beta2)*(grad.*grad);

    switch strategy
        case 1
            xn = xo - eta*grad;
        case 2
            xn = xo - vn;
        case 3
            xn = xo - (1+gamma)*vn + gamma*vo;
        case 4
            A = 1./sqrt(diagG + epsilon);
            A = eta*A;
            xn = xo - A.*grad;
        case 5
            A = sqrt(Eg2 + epsilon);
            A = eta./A;
            xn = xo - A.*grad;            
        case 6
            B = sqrt(Et2 + epsilon);
            A = sqrt(Eg2 + epsilon);
            C = B./A;
            xn = xo - C.*grad;
        case 7
            mtw = mt./(1-power(beta1,i));
            vtw = vt./(1-power(beta2,i));
            A = eta ./ (sqrt(vtw) + epsilon);
            xn = xo - A.*mtw;
        otherwise
            fprintf('error\n');    
    end
end
