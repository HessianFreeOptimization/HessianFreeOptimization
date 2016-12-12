function [ ] = plot_curve(records, fig)
    fig;
    subplot(1,2,1);
    max_llrecord = 0;
    llrecord_mean = zeros(size(records{1}.llrecord(:,1)));
    errrecord_mean = zeros(size(records{1}.errrecord(:,1)));
    
    for trial = 1:length(records)
        iters = size(records{trial}.llrecord, 1) - 1;
        if nargin == 2
            x_inds = 0:iters;
        else
            x_inds = eval_s;
        end
        trial_curve = semilogy(x_inds, -records{trial}.llrecord(:,1),'r-', 'LineWidth', 1.5);
%         alpha(trial_curve, 0.1);
        trial_curve.Color(4) = 0.2;
        llrecord_meanllrecord_mean = llrecord_mean + (-records{trial}.llrecord(:,1));
        hold on;
%         semilogy(x_inds, -records{trial}.llrecord(:,2),'b-');
%         hold on;
        if max(max(-records{trial}.llrecord)) > max_llrecord
            max_llrecord = max(max(-records{trial}.llrecord));
        end
    end
    llrecord_mean = llrecord_mean / length(records);
    semilogy(x_inds, -records{trial}.llrecord(:,1),'r-', 'LineWidth', 2);
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
        errrecord_mean = errrecord_mean + records{trial}.errrecord(:,1);
        hold on;
%         semilogy(x_inds, records{trial}.errrecord(:,2),'b-');
%         hold on;
    end
    grid on;
    title('train/test error');
    ylim([0 1]);
    
end