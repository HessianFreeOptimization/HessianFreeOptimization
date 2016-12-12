function [ ] = plot_curve(llrecord,  errrecord, eval_s)
    if nargin == 2
        x_inds = 0:iters;
    else
        x_inds = evals;
    end
    iters = size(llrecord, 1) - 1;
    fig = figure('color', [1 1 1]);
    subplot(1,2,1);
    semilogy(x_inds, -llrecord(:,1),'r-', 'MarkerSize', 3);
    ylim([0 max(-llrecord(:,1))]);
    hold on;
    semilogy(x_inds, -llrecord(:,2),'b-');
    grid on;
    ylim([0 max(-llrecord(:,2))]);
    title('log-likelihood');

    subplot(1,2,2);
    semilogy(x_inds, errrecord(:,1),'r-');
    hold on;
    semilogy(x_inds, errrecord(:,2),'b-');
    grid on;
    title('error');
    ylim([0 1]);
    set(fig, 'Position', [10, 10, 1000, 400]);

end