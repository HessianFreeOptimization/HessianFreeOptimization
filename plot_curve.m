function [ ] = plot_curve(records, fig)
    fig;
    subplot(1,2,1);
    max_llrecord = 0;
    for trial = 1:length(records)
        iters = size(records{trial}.llrecord, 1) - 1;
        if nargin == 2
            x_inds = 0:iters;
        else
            x_inds = eval_s;
        end
        semilogy(x_inds, -records{trial}.llrecord(:,1),'r-', 'MarkerSize', 3);
        hold on;
        semilogy(x_inds, -records{trial}.llrecord(:,2),'b-');
        hold on;
        if max(max(-records{trial}.llrecord)) > max_llrecord
            max_llrecord = max(max(-records{trial}.llrecord));
        end
    end
    grid on;
    max_llrecord
    ylim([0, max_llrecord]);
    title('objective');
    
    subplot(1,2,2);
    for trial = 1:length(records)
        iters = size(records{trial}.llrecord, 1) - 1;
        if nargin == 2
            x_inds = 0:iters;
        else
            x_inds = eval_s;
        end
        records{trial}.errrecord
        semilogy(x_inds, records{trial}.errrecord(:,1),'r-');
        hold on;
        semilogy(x_inds, records{trial}.errrecord(:,2),'b-');
        hold on;
    end
    grid on;
    title('train/test error');
    ylim([0 1]);
end