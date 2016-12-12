function [ ] = plot_curve(if_test, records, fig)
    fig;
    subplot(1,2,1);
    max_llrecord = 0;
    llrecord_mean = zeros(size(records{1}.llrecord(:,1)));
    errrecord_mean = zeros(size(records{1}.errrecord(:,1)));
    llrecord_mean_test = zeros(size(records{1}.llrecord(:,2)));
    errrecord_mean_test = zeros(size(records{1}.errrecord(:,2)));
    trials = length(records);
    
    for trial = 1:trials
        iters = size(records{trial}.llrecord, 1) - 1;
        x_inds = 0:iters;
%         x_inds = records{trial}.eval_gs;
        trial_curve = plot(x_inds, -records{trial}.llrecord(:,1),'r-', 'LineWidth', 1.5);
%         alpha(trial_curve, 0.1);
        trial_curve.Color(4) = 0.3;
        llrecord_mean = llrecord_mean + (-records{trial}.llrecord(:,1));
        hold on;
        if max(max(-records{trial}.llrecord)) > max_llrecord
            max_llrecord = max(max(-records{trial}.llrecord));
        end
        if if_test
            trial_curve_test = plot(x_inds, -records{trial}.llrecord(:,2),'b-');
            trial_curve_test.Color(4) = 0.3;
            llrecord_mean_test = llrecord_mean_test + (-records{trial}.llrecord(:,2));
            hold on;
        end
    end
    llrecord_mean = llrecord_mean / trials;
    plot(x_inds, llrecord_mean,'r-', 'LineWidth', 2.5); hold on;
    if if_test
        llrecord_mean_test = llrecord_mean_test / trials;
        plot(x_inds, llrecord_mean_test,'b-', 'LineWidth', 2.5);
    end
    grid on;
    ylim([0, max_llrecord]);
    title('objective');
    
    subplot(1,2,2);
    for trial = 1:trials
        iters = size(records{trial}.llrecord, 1) - 1;
        x_inds = 0:iters;
%         x_inds = records{trial}.eval_gs;
        trial_curve = plot(x_inds, records{trial}.errrecord(:,1),'r-', 'LineWidth', 1.5);
        trial_curve.Color(4) = 0.3;
        errrecord_mean = errrecord_mean + records{trial}.errrecord(:,1);
        hold on;
        if if_test
            trial_curve_test = plot(x_inds, records{trial}.errrecord(:,2),'b-');
            trial_curve_test.Color(4) = 0.3;
            errrecord_mean_test = errrecord_mean_test + (records{trial}.errrecord(:,2));
            hold on;
        end
    end
    errrecord_mean = errrecord_mean / trials;
    plot(x_inds, errrecord_mean,'r-', 'LineWidth', 2.5); hold on;
    if if_test
        errrecord_mean_test = errrecord_mean_test / trials;
        plot(x_inds, errrecord_mean_test,'b-', 'LineWidth', 2.5);
    end
    grid on;
    title('train/test error');
    ylim([0 1]);
    
end