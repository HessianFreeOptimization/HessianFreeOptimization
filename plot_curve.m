function [ ] = plot_curve( llrecord,  errrecord)
    iters = size(llrecord, 1) - 1;
    fig = figure('color', [1 1 1]);
    subplot(1,2,1);
    plot(0:iters, -llrecord(:,1),'r-', 'MarkerSize', 3);
    ylim([0 max(-llrecord(:,1))]);
    hold on;
    plot(0:iters, -llrecord(:,2),'b-');
    ylim([0 max(-llrecord(:,2))]);
    title('log-likelihood');

    subplot(1,2,2);
    plot(0:iters, errrecord(:,1),'r-');
    hold on;
    plot(0:iters, errrecord(:,2),'b-');
    title('error');
    ylim([0 1]);
    set(fig, 'Position', [10, 10, 1000, 400]);

end